import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
class LazyReferenceMapper(collections.abc.MutableMapping):
    """This interface can be used to read/write references from Parquet stores.
    It is not intended for other types of references.
    It can be used with Kerchunk's MultiZarrToZarr method to combine
    references into a parquet store.
    Examples of this use-case can be found here:
    https://fsspec.github.io/kerchunk/advanced.html?highlight=parquet#parquet-storage"""

    @property
    def np(self):
        import numpy as np
        return np

    @property
    def pd(self):
        import pandas as pd
        return pd

    def __init__(self, root, fs=None, out_root=None, cache_size=128, categorical_threshold=10):
        """

        This instance will be writable, storing changes in memory until full partitions
        are accumulated or .flush() is called.

        To create an empty lazy store, use .create()

        Parameters
        ----------
        root : str
            Root of parquet store
        fs : fsspec.AbstractFileSystem
            fsspec filesystem object, default is local filesystem.
        cache_size : int, default=128
            Maximum size of LRU cache, where cache_size*record_size denotes
            the total number of references that can be loaded in memory at once.
        categorical_threshold : int
            Encode urls as pandas.Categorical to reduce memory footprint if the ratio
            of the number of unique urls to total number of refs for each variable
            is greater than or equal to this number. (default 10)
        """
        self.root = root
        self.chunk_sizes = {}
        self.out_root = out_root or self.root
        self.cat_thresh = categorical_threshold
        self.cache_size = cache_size
        self.dirs = None
        self.url = self.root + '/{field}/refs.{record}.parq'
        self.fs = fsspec.filesystem('file') if fs is None else fs

    def __getattr__(self, item):
        if item in ('_items', 'record_size', 'zmetadata'):
            self.setup()
            return self.__dict__[item]
        raise AttributeError(item)

    def setup(self):
        self._items = {}
        self._items['.zmetadata'] = self.fs.cat_file('/'.join([self.root, '.zmetadata']))
        met = json.loads(self._items['.zmetadata'])
        self.record_size = met['record_size']
        self.zmetadata = met['metadata']

        @lru_cache(maxsize=self.cache_size)
        def open_refs(field, record):
            """cached parquet file loader"""
            path = self.url.format(field=field, record=record)
            data = io.BytesIO(self.fs.cat_file(path))
            df = self.pd.read_parquet(data, engine='fastparquet')
            refs = {c: df[c].values for c in df.columns}
            return refs
        self.open_refs = open_refs

    @staticmethod
    def create(root, storage_options=None, fs=None, record_size=10000, **kwargs):
        """Make empty parquet reference set

        First deletes the contents of the given directory, if it exists.

        Parameters
        ----------
        root: str
            Directory to contain the output; will be created
        storage_options: dict | None
            For making the filesystem to use for writing is fs is None
        fs: FileSystem | None
            Filesystem for writing
        record_size: int
            Number of references per parquet file
        kwargs: passed to __init__

        Returns
        -------
        LazyReferenceMapper instance
        """
        met = {'metadata': {}, 'record_size': record_size}
        if fs is None:
            fs, root = fsspec.core.url_to_fs(root, **storage_options or {})
        if fs.exists(root):
            fs.rm(root, recursive=True)
        fs.makedirs(root, exist_ok=True)
        fs.pipe('/'.join([root, '.zmetadata']), json.dumps(met).encode())
        return LazyReferenceMapper(root, fs, **kwargs)

    def listdir(self, basename=True):
        """List top-level directories"""
        if self.dirs is None:
            dirs = [p.split('/', 1)[0] for p in self.zmetadata]
            self.dirs = {p for p in dirs if p and (not p.startswith('.'))}
        listing = self.dirs
        if basename:
            listing = [os.path.basename(path) for path in listing]
        return listing

    def ls(self, path='', detail=True):
        """Shortcut file listings"""
        if not path:
            dirnames = self.listdir()
            others = set(['.zmetadata'] + [name for name in self.zmetadata if '/' not in name] + [name for name in self._items if '/' not in name])
            if detail is False:
                others.update(dirnames)
                return sorted(others)
            dirinfo = [{'name': name, 'type': 'directory', 'size': 0} for name in dirnames]
            fileinfo = [{'name': name, 'type': 'file', 'size': len(json.dumps(self.zmetadata[name]) if name in self.zmetadata else self._items[name])} for name in others]
            return sorted(dirinfo + fileinfo, key=lambda s: s['name'])
        parts = path.split('/', 1)
        if len(parts) > 1:
            raise FileNotFoundError('Cannot list within directories right now')
        field = parts[0]
        others = set([name for name in self.zmetadata if name.startswith(f'{path}/')] + [name for name in self._items if name.startswith(f'{path}/')])
        fileinfo = [{'name': name, 'type': 'file', 'size': len(json.dumps(self.zmetadata[name]) if name in self.zmetadata else self._items[name])} for name in others]
        keys = self._keys_in_field(field)
        if detail is False:
            return list(others) + list(keys)
        recs = self._generate_all_records(field)
        recinfo = [{'name': name, 'type': 'file', 'size': rec[-1]} for name, rec in zip(keys, recs) if rec[0]]
        return fileinfo + recinfo

    def _load_one_key(self, key):
        """Get the reference for one key

        Returns bytes, one-element list or three-element list.
        """
        if key in self._items:
            return self._items[key]
        elif key in self.zmetadata:
            return json.dumps(self.zmetadata[key]).encode()
        elif '/' not in key or self._is_meta(key):
            raise KeyError(key)
        field, _ = key.rsplit('/', 1)
        record, ri, chunk_size = self._key_to_record(key)
        maybe = self._items.get((field, record), {}).get(ri, False)
        if maybe is None:
            raise KeyError
        elif maybe:
            return maybe
        elif chunk_size == 0:
            return b''
        try:
            refs = self.open_refs(field, record)
        except (ValueError, TypeError, FileNotFoundError):
            raise KeyError(key)
        columns = ['path', 'offset', 'size', 'raw']
        selection = [refs[c][ri] if c in refs else None for c in columns]
        raw = selection[-1]
        if raw is not None:
            return raw
        if selection[0] is None:
            raise KeyError('This reference does not exist or has been deleted')
        if selection[1:3] == [0, 0]:
            return selection[:1]
        return selection[:3]

    @lru_cache(4096)
    def _key_to_record(self, key):
        """Details needed to construct a reference for one key"""
        field, chunk = key.rsplit('/', 1)
        chunk_sizes = self._get_chunk_sizes(field)
        if len(chunk_sizes) == 0:
            return (0, 0, 0)
        chunk_idx = [int(c) for c in chunk.split('.')]
        chunk_number = ravel_multi_index(chunk_idx, chunk_sizes)
        record = chunk_number // self.record_size
        ri = chunk_number % self.record_size
        return (record, ri, len(chunk_sizes))

    def _get_chunk_sizes(self, field):
        """The number of chunks along each axis for a given field"""
        if field not in self.chunk_sizes:
            zarray = self.zmetadata[f'{field}/.zarray']
            size_ratio = [math.ceil(s / c) for s, c in zip(zarray['shape'], zarray['chunks'])]
            self.chunk_sizes[field] = size_ratio or [1]
        return self.chunk_sizes[field]

    def _generate_record(self, field, record):
        """The references for a given parquet file of a given field"""
        refs = self.open_refs(field, record)
        it = iter(zip(*refs.values()))
        if len(refs) == 3:
            return (list(t) for t in it)
        elif len(refs) == 1:
            return refs['raw']
        else:
            return (list(t[:3]) if not t[3] else t[3] for t in it)

    def _generate_all_records(self, field):
        """Load all the references within a field by iterating over the parquet files"""
        nrec = 1
        for ch in self._get_chunk_sizes(field):
            nrec *= ch
        nrec = math.ceil(nrec / self.record_size)
        for record in range(nrec):
            yield from self._generate_record(field, record)

    def values(self):
        return RefsValuesView(self)

    def items(self):
        return RefsItemsView(self)

    def __hash__(self):
        return id(self)

    def __getitem__(self, key):
        return self._load_one_key(key)

    def __setitem__(self, key, value):
        if '/' in key and (not self._is_meta(key)):
            field, chunk = key.rsplit('/', 1)
            record, i, _ = self._key_to_record(key)
            subdict = self._items.setdefault((field, record), {})
            subdict[i] = value
            if len(subdict) == self.record_size:
                self.write(field, record)
        else:
            self._items[key] = value
            new_value = json.loads(value.decode() if isinstance(value, bytes) else value)
            self.zmetadata[key] = {**self.zmetadata.get(key, {}), **new_value}

    @staticmethod
    def _is_meta(key):
        return key.startswith('.z') or '/.z' in key

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        elif key in self.zmetadata:
            del self.zmetadata[key]
        elif '/' in key and (not self._is_meta(key)):
            field, _ = key.rsplit('/', 1)
            record, i, _ = self._key_to_record(key)
            subdict = self._items.setdefault((field, record), {})
            subdict[i] = None
            if len(subdict) == self.record_size:
                self.write(field, record)
        else:
            self._items[key] = None

    def write(self, field, record, base_url=None, storage_options=None):
        import kerchunk.df
        import numpy as np
        import pandas as pd
        partition = self._items[field, record]
        original = False
        if len(partition) < self.record_size:
            try:
                original = self.open_refs(field, record)
            except IOError:
                pass
        if original:
            paths = original['path']
            offsets = original['offset']
            sizes = original['size']
            raws = original['raw']
        else:
            paths = np.full(self.record_size, np.nan, dtype='O')
            offsets = np.zeros(self.record_size, dtype='int64')
            sizes = np.zeros(self.record_size, dtype='int64')
            raws = np.full(self.record_size, np.nan, dtype='O')
        for j, data in partition.items():
            if isinstance(data, list):
                if str(paths.dtype) == 'category' and data[0] not in paths.dtype.categories:
                    paths = paths.add_categories(data[0])
                paths[j] = data[0]
                if len(data) > 1:
                    offsets[j] = data[1]
                    sizes[j] = data[2]
            elif data is None:
                paths[j] = None
                offsets[j] = 0
                sizes[j] = 0
                raws[j] = None
            else:
                raws[j] = kerchunk.df._proc_raw(data)
        df = pd.DataFrame({'path': paths, 'offset': offsets, 'size': sizes, 'raw': raws}, copy=False)
        if df.path.count() / (df.path.nunique() or 1) > self.cat_thresh:
            df['path'] = df['path'].astype('category')
        object_encoding = {'raw': 'bytes', 'path': 'utf8'}
        has_nulls = ['path', 'raw']
        fn = f'{base_url or self.out_root}/{field}/refs.{record}.parq'
        self.fs.mkdirs(f'{base_url or self.out_root}/{field}', exist_ok=True)
        df.to_parquet(fn, engine='fastparquet', storage_options=storage_options or getattr(self.fs, 'storage_options', None), compression='zstd', index=False, stats=False, object_encoding=object_encoding, has_nulls=has_nulls)
        partition.clear()
        self._items.pop((field, record))

    def flush(self, base_url=None, storage_options=None):
        """Output any modified or deleted keys

        Parameters
        ----------
        base_url: str
            Location of the output
        """
        for thing in list(self._items):
            if isinstance(thing, tuple):
                field, record = thing
                self.write(field, record, base_url=base_url, storage_options=storage_options)
        for k in list(self._items):
            if k != '.zmetadata' and '.z' in k:
                self.zmetadata[k] = json.loads(self._items.pop(k))
        met = {'metadata': self.zmetadata, 'record_size': self.record_size}
        self._items['.zmetadata'] = json.dumps(met).encode()
        self.fs.pipe('/'.join([base_url or self.out_root, '.zmetadata']), self._items['.zmetadata'])
        self.open_refs.cache_clear()

    def __len__(self):
        count = 0
        for field in self.listdir():
            if field.startswith('.'):
                count += 1
            else:
                count += math.prod(self._get_chunk_sizes(field))
        count += len(self.zmetadata)
        count += sum((1 for _ in self._items if not isinstance(_, tuple)))
        return count

    def __iter__(self):
        metas = set(self.zmetadata)
        metas.update(self._items)
        for bit in metas:
            if isinstance(bit, str):
                yield bit
        for field in self.listdir():
            for k in self._keys_in_field(field):
                if k in self:
                    yield k

    def __contains__(self, item):
        try:
            self._load_one_key(item)
            return True
        except KeyError:
            return False

    def _keys_in_field(self, field):
        """List key names in given field

        Produces strings like "field/x.y" appropriate from the chunking of the array
        """
        chunk_sizes = self._get_chunk_sizes(field)
        if len(chunk_sizes) == 0:
            yield (field + '/0')
            return
        inds = itertools.product(*(range(i) for i in chunk_sizes))
        for ind in inds:
            yield (field + '/' + '.'.join([str(c) for c in ind]))