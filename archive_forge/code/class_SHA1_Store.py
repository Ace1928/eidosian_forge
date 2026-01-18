import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union, cast
import pgzip
import torch
from torch import Tensor
from fairscale.internal.containers import from_np, to_np
from .utils import ExitCode
class SHA1_Store:
    """
    This class represents a SHA1 checksum based storage dir for state_dict
    and tensors.

    This means the same content will not be stored multiple times, resulting
    in space savings. (a.k.a. de-duplication)

    To make things easier for the callers, this class accept input data
    as files, state_dict or tensors. This class always returns in-memory
    data, not on-disk files. This class doesn't really care or know the actually
    data types.

    A key issue is dealing with content deletion. We use a reference counting
    algorithm, which means the caller must have symmetrical add/remove calls
    for each object.

    We used to support children-parent dependency graph and ref counting, but
    it is flawed since a grand-child can have the same SHA1 as the grand-parent,
    resulting in a cycle. This means caller must compute which parent is safe
    to delete in a version tracking graph. The lesson here is that content
    addressibility and dependency graphs do not mix well.

    We support multicore compression for the data to be store on per-object basis.
    We use pgzip to do parallel compression/decompression on top of it to use all
    the cores.

    Args:
        path (Path):
            The path in which a SHA1_Store will be created.
        init (bool, optional):
            - If ``True``, a new SHA1_Store in the path if not already exists.
            - Default: False
        sha1_buf_size (int):
            Buffer size used for checksumming. Default: 100MB.
        tmp_dir (str):
            Dir for temporary files if input is an in-memory object or output data needs
            to be decompressed first.
        pgzip_threads (int, optional):
            Number of threads (cores) used in compression. Default: None to use all cores.
        pgzip_block_size (int):
            Per-thread block size for compression. Default: 10MB.
    """

    def __init__(self, path: Path, init: bool=False, sha1_buf_size: int=100 * 1024 * 1024, tmp_dir: str='', pgzip_threads: Optional[int]=None, pgzip_block_size: int=10 * 1024 * 1024) -> None:
        """Create or wrap (if already exists) a store."""
        self._path = path
        self._sha1_buf_size = sha1_buf_size
        self._pgzip_threads = pgzip_threads
        self._pgzip_block_size = pgzip_block_size
        self._metadata_file_path = self._path.joinpath('metadata.json')
        self._json_dict: Optional[Dict[str, Any]] = None
        self._json_ctx = _JSON_DictContext(self, readonly=False)
        self._readonly_json_ctx = _JSON_DictContext(self, readonly=True)
        if init and (not self._path.exists()):
            try:
                Path.mkdir(self._path, parents=False, exist_ok=False)
            except FileExistsError as error:
                sys.stderr.write(f'An exception occured while creating Sha1_store: {repr(error)}\n')
                sys.exit(ExitCode.FILE_EXISTS_ERROR)
            with self._json_ctx:
                self._json_dict = {STORE_CREATE_DATE_KEY: time.ctime(), STORE_OS_KEY: 0, STORE_DS_KEY: 0, STORE_CS_KEY: 0}
        assert self._path.exists() and self._metadata_file_path.exists(), f'SHA1 store {self._path} does not exist and init is False'
        with self._readonly_json_ctx:
            assert STORE_CREATE_DATE_KEY in self._json_dict, f'Invalid SHA1 Store in {self._path}'
        if tmp_dir:
            assert Path(tmp_dir).is_dir(), 'incorrect input'
            self._tmp_dir = Path(tmp_dir)
        else:
            self._tmp_dir = self._path.joinpath('tmp')
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir.mkdir()

    def add(self, file_or_obj: Union[Path, Tensor, Dict], compress: bool=True, name: str=None) -> str:
        """Adds a file/object to this store and the sha1 references accordingly.

        First, a sha1 hash is calculated. Utilizing the sha1 hash string, the actual file
        in <file_or_obj> is moved within the store and the reference file is updated.
        If the input is an object, it will be store in the self._tmp_dir and then moved.

        If compress is True, the stored file is also compressed, which is useful for tensors
        with a lot of zeros.

        We use pickle and numpy for saving, loading because it is more deterministic
        in terms of serialized bytes. They do lose info on device and dtype of
        tensors. Will handle those later.

        Args:
            file_or_obj (str or tensor or Dict):
                Path to the file to be added to the store or an in-memory object
                that can be handled by pickle. Note, OrderedDict is used when
                you call `state_dict()` on a nn.Module, and it is an instance
                of a Dict too. A model's state_dict can be a simple dict because
                it may contain both model state_dict and other non-tensor info.
            compress (bool, optional):
                Use gzip compression on this object or not.
                Default: True
            name (str, optional):
                Optional name for this object.
                Default: None
        """
        start = time.time()
        is_pickle_file = None
        if isinstance(file_or_obj, (Path, str)):
            try:
                pickle.load(open(file_or_obj, 'rb'))
                is_pickle_file = True
            except Exception as e:
                is_pickle_file = False
                pass
            file_path = Path(file_or_obj)
            remove_tmp = False
        if is_pickle_file is False:
            file_or_obj = torch.load(cast(Union[Path, str], file_or_obj))
        if isinstance(file_or_obj, (Tensor, Dict)):
            file_path = self._get_tmp_file_path()
            pickle.dump(to_np(file_or_obj), open(file_path, 'wb'))
            remove_tmp = True
        else:
            assert False, f'incorrect input {type(file_or_obj)}'
        assert isinstance(file_path, Path), type(file_path)
        sha1_hash = self._get_sha1_hash(file_path)
        with self._json_ctx:
            ref_count = self._add_ref(sha1_hash, True, compress)
            if ref_count == 1:
                repo_fdir = self._sha1_to_dir(sha1_hash)
                if not repo_fdir.exists():
                    try:
                        repo_fdir.mkdir(exist_ok=True, parents=True)
                    except FileExistsError as error:
                        sys.stderr.write(f'An exception occured: {repr(error)}\n')
                        sys.exit(ExitCode.FILE_EXISTS_ERROR)
                repo_fpath = repo_fdir.joinpath(sha1_hash)
                try:
                    if compress:
                        orig_size, comp_size = _copy_compressed(file_path, repo_fpath, self._pgzip_threads, self._pgzip_block_size)
                    else:
                        shutil.copy2(file_path, repo_fpath)
                        orig_size = comp_size = file_path.stat().st_size
                except BaseException as error:
                    sys.stderr.write(f'An exception occured: {repr(error)}\n')
                    ref_count = self._add_ref(sha1_hash, False, compress)
            entry = _get_json_entry(self._json_dict[sha1_hash])
            assert ref_count == 1 or entry[ENTRY_OS_KEY] % (ref_count - 1) == 0, f'incorrect size: {entry[ENTRY_OS_KEY]} and {ref_count}'
            o_diff = orig_size if ref_count == 1 else entry[ENTRY_OS_KEY] // (ref_count - 1)
            d_diff = orig_size if ref_count == 1 else 0
            c_diff = comp_size if ref_count == 1 else 0
            entry[ENTRY_OS_KEY] += o_diff
            entry[ENTRY_DS_KEY] += d_diff
            entry[ENTRY_CS_KEY] += c_diff
            self._json_dict[STORE_OS_KEY] += o_diff
            self._json_dict[STORE_DS_KEY] += d_diff
            self._json_dict[STORE_CS_KEY] += c_diff
            if name:
                if name not in entry[ENTRY_NAMES_KEY].keys():
                    entry[ENTRY_NAMES_KEY][name] = 1
                else:
                    entry[ENTRY_NAMES_KEY][name] += 1
        if remove_tmp:
            file_path.unlink()
        duration = time.time() - start
        if duration > 60:
            logging.warning(f'Add() is taking long: {duration}s')
        return sha1_hash

    def get(self, sha1: str) -> Union[Tensor, Dict]:
        """Get data from a SHA1

        Args:
            sha1 (str):
                SHA1 of the object to get.

        Returns:
            (Tensor or Dict):
                In-memory object.

        Throws:
            ValueError if sha1 is not found.
        """
        path = self._sha1_to_dir(sha1).joinpath(sha1)
        if not path.exists():
            raise ValueError(f'Try to get SHA1 {sha1} but it is not found')
        with self._readonly_json_ctx:
            if self._json_dict[sha1][ENTRY_COMP_KEY]:
                tmp = self._get_tmp_file_path()
                _copy_uncompressed(path, tmp, self._pgzip_threads, self._pgzip_block_size)
                obj = pickle.load(open(tmp, 'rb'))
                tmp.unlink()
            else:
                obj = pickle.load(open(path, 'rb'))
        return from_np(obj)

    def delete(self, sha1: str) -> None:
        """Delete a SHA1

        Args:
            sha1 (str):
                SHA1 of the object to delete.

        Throws:
            ValueError if sha1 is not found.
        """
        path = self._sha1_to_dir(sha1).joinpath(sha1)
        if not path.exists():
            raise ValueError(f'Try to delete SHA1 {sha1} but it is not found')
        with self._json_ctx:
            assert sha1 in self._json_dict.keys(), 'internal error: sha1 not found in json'
            entry = _get_json_entry(self._json_dict[sha1])
            assert entry[ENTRY_RF_KEY] > 0, f'ref count {entry[ENTRY_RF_KEY]} should be positive'
            entry[ENTRY_RF_KEY] -= 1
            if entry[ENTRY_RF_KEY] == 0:
                path.unlink()
                entry = {}
            if entry:
                self._json_dict[sha1] = entry
            else:
                del self._json_dict[sha1]

    def size_info(self, sha1: Optional[str]=None) -> Tuple[int, int, int]:
        """Return original, deduped, gzipped sizes for an entry or the store."""
        with self._readonly_json_ctx:
            if sha1:
                if sha1 not in self._json_dict.keys():
                    raise ValueError(f'SHA1 {sha1} not found')
                entry = self._json_dict[sha1]
                return (entry[ENTRY_OS_KEY], entry[ENTRY_DS_KEY], entry[ENTRY_CS_KEY])
            return (self._json_dict[STORE_OS_KEY], self._json_dict[STORE_DS_KEY], self._json_dict[STORE_CS_KEY])

    def names(self, sha1: str=None) -> Dict[str, int]:
        """Return the names dict for an object."""
        with self._readonly_json_ctx:
            if sha1 not in self._json_dict.keys():
                raise ValueError(f'SHA1 {sha1} not found')
            entry = self._json_dict[sha1]
            return entry[ENTRY_NAMES_KEY]

    def _get_sha1_hash(self, file_path: Union[str, Path]) -> str:
        """Return the sha1 hash of a file

        Args:
            file_path (str, Path):
                Path to the file whose sha1 hash is to be calculalated and returned.

        Returns:
            (str):
                The SHA1 computed.
        """
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(self._sha1_buf_size)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()

    def _get_tmp_file_path(self) -> Path:
        """Helper to get a tmp file name under self.tmp_dir."""
        fd, name = tempfile.mkstemp(dir=self._tmp_dir)
        os.close(fd)
        return Path(name)

    def _sha1_to_dir(self, sha1: str) -> Path:
        """Helper to get the internal dir for a file based on its SHA1"""
        assert len(sha1) > 4, 'sha1 too short'
        part1, part2 = (sha1[:2], sha1[2:4])
        return self._path.joinpath(part1, part2)

    def _add_ref(self, current_sha1_hash: str, inc: bool, compressed: bool) -> int:
        """
        Update the reference count.

        If the reference counting file does not have this sha1, then a new tracking
        entry of the added.

        Args:
            current_sha1_hash (str):
                The sha1 hash of the incoming added file.
            inc (bool):
                Increment or decrement.

        Returns:
            (int):
                Resulting ref count.
        """
        if current_sha1_hash not in self._json_dict:
            entry = {}
        else:
            entry = self._json_dict[current_sha1_hash]
        entry = _get_json_entry(entry)
        entry[ENTRY_RF_KEY] += 1 if inc else -1
        assert entry[ENTRY_RF_KEY] >= 0, 'negative ref count'
        entry[ENTRY_COMP_KEY] = compressed
        self._json_dict[current_sha1_hash] = entry
        return entry[ENTRY_RF_KEY]