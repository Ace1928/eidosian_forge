import hashlib
import os
import site
import sys
import tarfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from wasabi import msg
from ..errors import Errors
from ..util import check_spacy_env_vars, download_file, ensure_pathy, get_checksum
from ..util import get_hash, make_tempdir, upload_file
class RemoteStorage:
    """Push and pull outputs to and from a remote file storage.

    Remotes can be anything that `smart-open` can support: AWS, GCS, file system,
    ssh, etc.
    """

    def __init__(self, project_root: Path, url: str, *, compression='gz'):
        self.root = project_root
        self.url = ensure_pathy(url)
        self.compression = compression

    def push(self, path: Path, command_hash: str, content_hash: str) -> 'CloudPath':
        """Compress a file or directory within a project and upload it to a remote
        storage. If an object exists at the full URL, nothing is done.

        Within the remote storage, files are addressed by their project path
        (url encoded) and two user-supplied hashes, representing their creation
        context and their file contents. If the URL already exists, the data is
        not uploaded. Paths are archived and compressed prior to upload.
        """
        loc = self.root / path
        if not loc.exists():
            raise IOError(f'Cannot push {loc}: does not exist.')
        url = self.make_url(path, command_hash, content_hash)
        if url.exists():
            return url
        tmp: Path
        with make_tempdir() as tmp:
            tar_loc = tmp / self.encode_name(str(path))
            mode_string = f'w:{self.compression}' if self.compression else 'w'
            with tarfile.open(tar_loc, mode=mode_string) as tar_file:
                tar_file.add(str(loc), arcname=str(path))
            upload_file(tar_loc, url)
        return url

    def pull(self, path: Path, *, command_hash: Optional[str]=None, content_hash: Optional[str]=None) -> Optional['CloudPath']:
        """Retrieve a file from the remote cache. If the file already exists,
        nothing is done.

        If the command_hash and/or content_hash are specified, only matching
        results are returned. If no results are available, an error is raised.
        """
        dest = self.root / path
        if dest.exists():
            return None
        url = self.find(path, command_hash=command_hash, content_hash=content_hash)
        if url is None:
            return url
        else:
            if not dest.parent.exists():
                dest.parent.mkdir(parents=True)
            tmp: Path
            with make_tempdir() as tmp:
                tar_loc = tmp / url.parts[-1]
                download_file(url, tar_loc)
                mode_string = f'r:{self.compression}' if self.compression else 'r'
                with tarfile.open(tar_loc, mode=mode_string) as tar_file:

                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        return prefix == abs_directory

                    def safe_extract(tar, path):
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise ValueError(Errors.E201)
                        if sys.version_info >= (3, 12):
                            tar.extractall(path, filter='data')
                        else:
                            tar.extractall(path)
                    safe_extract(tar_file, self.root)
        return url

    def find(self, path: Path, *, command_hash: Optional[str]=None, content_hash: Optional[str]=None) -> Optional['CloudPath']:
        """Find the best matching version of a file within the storage,
        or `None` if no match can be found. If both the creation and content hash
        are specified, only exact matches will be returned. Otherwise, the most
        recent matching file is preferred.
        """
        name = self.encode_name(str(path))
        urls = []
        if command_hash is not None and content_hash is not None:
            url = self.url / name / command_hash / content_hash
            urls = [url] if url.exists() else []
        elif command_hash is not None:
            if (self.url / name / command_hash).exists():
                urls = list((self.url / name / command_hash).iterdir())
        elif (self.url / name).exists():
            for sub_dir in (self.url / name).iterdir():
                urls.extend(sub_dir.iterdir())
            if content_hash is not None:
                urls = [url for url in urls if url.parts[-1] == content_hash]
        if len(urls) >= 2:
            try:
                urls.sort(key=lambda x: x.stat().st_mtime)
            except Exception:
                msg.warn('Unable to sort remote files by last modified. The file(s) pulled from the cache may not be the most recent.')
        return urls[-1] if urls else None

    def make_url(self, path: Path, command_hash: str, content_hash: str) -> 'CloudPath':
        """Construct a URL from a subpath, a creation hash and a content hash."""
        return self.url / self.encode_name(str(path)) / command_hash / content_hash

    def encode_name(self, name: str) -> str:
        """Encode a subpath into a URL-safe name."""
        return urllib.parse.quote_plus(name)