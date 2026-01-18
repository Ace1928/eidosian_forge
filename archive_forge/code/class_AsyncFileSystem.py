import asyncio
import asyncio.events
import functools
import inspect
import io
import numbers
import os
import re
import threading
from contextlib import contextmanager
from glob import has_magic
from typing import TYPE_CHECKING, Iterable
from .callbacks import DEFAULT_CALLBACK
from .exceptions import FSTimeoutError
from .implementations.local import LocalFileSystem, make_path_posix, trailing_sep
from .spec import AbstractBufferedFile, AbstractFileSystem
from .utils import glob_translate, is_exception, other_paths
class AsyncFileSystem(AbstractFileSystem):
    """Async file operations, default implementations

    Passes bulk operations to asyncio.gather for concurrent operation.

    Implementations that have concurrent batch operations and/or async methods
    should inherit from this class instead of AbstractFileSystem. Docstrings are
    copied from the un-underscored method in AbstractFileSystem, if not given.
    """
    async_impl = True
    mirror_sync_methods = True
    disable_throttling = False

    def __init__(self, *args, asynchronous=False, loop=None, batch_size=None, **kwargs):
        self.asynchronous = asynchronous
        self._pid = os.getpid()
        if not asynchronous:
            self._loop = loop or get_loop()
        else:
            self._loop = None
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    @property
    def loop(self):
        if self._pid != os.getpid():
            raise RuntimeError('This class is not fork-safe')
        return self._loop

    async def _rm_file(self, path, **kwargs):
        raise NotImplementedError

    async def _rm(self, path, recursive=False, batch_size=None, **kwargs):
        batch_size = batch_size or self.batch_size
        path = await self._expand_path(path, recursive=recursive)
        return await _run_coros_in_chunks([self._rm_file(p, **kwargs) for p in reversed(path)], batch_size=batch_size, nofiles=True)

    async def _cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError

    async def _copy(self, path1, path2, recursive=False, on_error=None, maxdepth=None, batch_size=None, **kwargs):
        if on_error is None and recursive:
            on_error = 'ignore'
        elif on_error is None:
            on_error = 'raise'
        if isinstance(path1, list) and isinstance(path2, list):
            paths1 = path1
            paths2 = path2
        else:
            source_is_str = isinstance(path1, str)
            paths1 = await self._expand_path(path1, maxdepth=maxdepth, recursive=recursive)
            if source_is_str and (not recursive or maxdepth is not None):
                paths1 = [p for p in paths1 if not (trailing_sep(p) or await self._isdir(p))]
                if not paths1:
                    return
            source_is_file = len(paths1) == 1
            dest_is_dir = isinstance(path2, str) and (trailing_sep(path2) or await self._isdir(path2))
            exists = source_is_str and (has_magic(path1) and source_is_file or (not has_magic(path1) and dest_is_dir and (not trailing_sep(path1))))
            paths2 = other_paths(paths1, path2, exists=exists, flatten=not source_is_str)
        batch_size = batch_size or self.batch_size
        coros = [self._cp_file(p1, p2, **kwargs) for p1, p2 in zip(paths1, paths2)]
        result = await _run_coros_in_chunks(coros, batch_size=batch_size, return_exceptions=True, nofiles=True)
        for ex in filter(is_exception, result):
            if on_error == 'ignore' and isinstance(ex, FileNotFoundError):
                continue
            raise ex

    async def _pipe_file(self, path, value, **kwargs):
        raise NotImplementedError

    async def _pipe(self, path, value=None, batch_size=None, **kwargs):
        if isinstance(path, str):
            path = {path: value}
        batch_size = batch_size or self.batch_size
        return await _run_coros_in_chunks([self._pipe_file(k, v, **kwargs) for k, v in path.items()], batch_size=batch_size, nofiles=True)

    async def _process_limits(self, url, start, end):
        """Helper for "Range"-based _cat_file"""
        size = None
        suff = False
        if start is not None and start < 0:
            if end is None:
                end = -start
                start = ''
                suff = True
            else:
                size = size or (await self._info(url))['size']
                start = size + start
        elif start is None:
            start = 0
        if not suff:
            if end is not None and end < 0:
                if start is not None:
                    size = size or (await self._info(url))['size']
                    end = size + end
            elif end is None:
                end = ''
            if isinstance(end, numbers.Integral):
                end -= 1
        return f'bytes={start}-{end}'

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        raise NotImplementedError

    async def _cat(self, path, recursive=False, on_error='raise', batch_size=None, **kwargs):
        paths = await self._expand_path(path, recursive=recursive)
        coros = [self._cat_file(path, **kwargs) for path in paths]
        batch_size = batch_size or self.batch_size
        out = await _run_coros_in_chunks(coros, batch_size=batch_size, nofiles=True, return_exceptions=True)
        if on_error == 'raise':
            ex = next(filter(is_exception, out), False)
            if ex:
                raise ex
        if len(paths) > 1 or isinstance(path, list) or paths[0] != self._strip_protocol(path):
            return {k: v for k, v in zip(paths, out) if on_error != 'omit' or not is_exception(v)}
        else:
            return out[0]

    async def _cat_ranges(self, paths, starts, ends, max_gap=None, batch_size=None, on_error='return', **kwargs):
        """Get the contents of byte ranges from one or more files

        Parameters
        ----------
        paths: list
            A list of of filepaths on this filesystems
        starts, ends: int or list
            Bytes limits of the read. If using a single int, the same value will be
            used to read all the specified files.
        """
        if max_gap is not None:
            raise NotImplementedError
        if not isinstance(paths, list):
            raise TypeError
        if not isinstance(starts, Iterable):
            starts = [starts] * len(paths)
        if not isinstance(ends, Iterable):
            ends = [ends] * len(paths)
        if len(starts) != len(paths) or len(ends) != len(paths):
            raise ValueError
        coros = [self._cat_file(p, start=s, end=e, **kwargs) for p, s, e in zip(paths, starts, ends)]
        batch_size = batch_size or self.batch_size
        return await _run_coros_in_chunks(coros, batch_size=batch_size, nofiles=True, return_exceptions=True)

    async def _put_file(self, lpath, rpath, **kwargs):
        raise NotImplementedError

    async def _put(self, lpath, rpath, recursive=False, callback=DEFAULT_CALLBACK, batch_size=None, maxdepth=None, **kwargs):
        """Copy file(s) from local.

        Copies a specific file or tree of files (if recursive=True). If rpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within.

        The put_file method will be called concurrently on a batch of files. The
        batch_size option can configure the amount of futures that can be executed
        at the same time. If it is -1, then all the files will be uploaded concurrently.
        The default can be set for this instance by passing "batch_size" in the
        constructor, or for all instances by setting the "gather_batch_size" key
        in ``fsspec.config.conf``, falling back to 1/8th of the system limit .
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            rpaths = rpath
            lpaths = lpath
        else:
            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(lpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                lpaths = [p for p in lpaths if not (trailing_sep(p) or fs.isdir(p))]
                if not lpaths:
                    return
            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (trailing_sep(rpath) or await self._isdir(rpath))
            rpath = self._strip_protocol(rpath)
            exists = source_is_str and (has_magic(lpath) and source_is_file or (not has_magic(lpath) and dest_is_dir and (not trailing_sep(lpath))))
            rpaths = other_paths(lpaths, rpath, exists=exists, flatten=not source_is_str)
        is_dir = {l: os.path.isdir(l) for l in lpaths}
        rdirs = [r for l, r in zip(lpaths, rpaths) if is_dir[l]]
        file_pairs = [(l, r) for l, r in zip(lpaths, rpaths) if not is_dir[l]]
        await asyncio.gather(*[self._makedirs(d, exist_ok=True) for d in rdirs])
        batch_size = batch_size or self.batch_size
        coros = []
        callback.set_size(len(file_pairs))
        for lfile, rfile in file_pairs:
            put_file = callback.branch_coro(self._put_file)
            coros.append(put_file(lfile, rfile, **kwargs))
        return await _run_coros_in_chunks(coros, batch_size=batch_size, callback=callback)

    async def _get_file(self, rpath, lpath, **kwargs):
        raise NotImplementedError

    async def _get(self, rpath, lpath, recursive=False, callback=DEFAULT_CALLBACK, maxdepth=None, **kwargs):
        """Copy file(s) to local.

        Copies a specific file or tree of files (if recursive=True). If lpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within. Can submit a list of paths, which may be glob-patterns
        and will be expanded.

        The get_file method will be called concurrently on a batch of files. The
        batch_size option can configure the amount of futures that can be executed
        at the same time. If it is -1, then all the files will be uploaded concurrently.
        The default can be set for this instance by passing "batch_size" in the
        constructor, or for all instances by setting the "gather_batch_size" key
        in ``fsspec.config.conf``, falling back to 1/8th of the system limit .
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            rpaths = rpath
            lpaths = lpath
        else:
            source_is_str = isinstance(rpath, str)
            source_not_trailing_sep = source_is_str and (not trailing_sep(rpath))
            rpath = self._strip_protocol(rpath)
            rpaths = await self._expand_path(rpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                rpaths = [p for p in rpaths if not (trailing_sep(p) or await self._isdir(p))]
                if not rpaths:
                    return
            lpath = make_path_posix(lpath)
            source_is_file = len(rpaths) == 1
            dest_is_dir = isinstance(lpath, str) and (trailing_sep(lpath) or LocalFileSystem().isdir(lpath))
            exists = source_is_str and (has_magic(rpath) and source_is_file or (not has_magic(rpath) and dest_is_dir and source_not_trailing_sep))
            lpaths = other_paths(rpaths, lpath, exists=exists, flatten=not source_is_str)
        [os.makedirs(os.path.dirname(lp), exist_ok=True) for lp in lpaths]
        batch_size = kwargs.pop('batch_size', self.batch_size)
        coros = []
        callback.set_size(len(lpaths))
        for lpath, rpath in zip(lpaths, rpaths):
            get_file = callback.branch_coro(self._get_file)
            coros.append(get_file(rpath, lpath, **kwargs))
        return await _run_coros_in_chunks(coros, batch_size=batch_size, callback=callback)

    async def _isfile(self, path):
        try:
            return (await self._info(path))['type'] == 'file'
        except:
            return False

    async def _isdir(self, path):
        try:
            return (await self._info(path))['type'] == 'directory'
        except OSError:
            return False

    async def _size(self, path):
        return (await self._info(path)).get('size', None)

    async def _sizes(self, paths, batch_size=None):
        batch_size = batch_size or self.batch_size
        return await _run_coros_in_chunks([self._size(p) for p in paths], batch_size=batch_size)

    async def _exists(self, path, **kwargs):
        try:
            await self._info(path, **kwargs)
            return True
        except FileNotFoundError:
            return False

    async def _info(self, path, **kwargs):
        raise NotImplementedError

    async def _ls(self, path, detail=True, **kwargs):
        raise NotImplementedError

    async def _walk(self, path, maxdepth=None, on_error='omit', **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        path = self._strip_protocol(path)
        full_dirs = {}
        dirs = {}
        files = {}
        detail = kwargs.pop('detail', False)
        try:
            listing = await self._ls(path, detail=True, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if on_error == 'raise':
                raise
            elif callable(on_error):
                on_error(e)
            if detail:
                yield (path, {}, {})
            else:
                yield (path, [], [])
            return
        for info in listing:
            pathname = info['name'].rstrip('/')
            name = pathname.rsplit('/', 1)[-1]
            if info['type'] == 'directory' and pathname != path:
                full_dirs[name] = pathname
                dirs[name] = info
            elif pathname == path:
                files[''] = info
            else:
                files[name] = info
        if detail:
            yield (path, dirs, files)
        else:
            yield (path, list(dirs), list(files))
        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                return
        for d in dirs:
            async for _ in self._walk(full_dirs[d], maxdepth=maxdepth, detail=detail, **kwargs):
                yield _

    async def _glob(self, path, maxdepth=None, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        import re
        seps = (os.path.sep, os.path.altsep) if os.path.altsep else (os.path.sep,)
        ends_with_sep = path.endswith(seps)
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_sep or path.endswith(tuple((sep + '**' for sep in seps)))
        idx_star = path.find('*') if path.find('*') >= 0 else len(path)
        idx_qmark = path.find('?') if path.find('?') >= 0 else len(path)
        idx_brace = path.find('[') if path.find('[') >= 0 else len(path)
        min_idx = min(idx_star, idx_qmark, idx_brace)
        detail = kwargs.pop('detail', False)
        if not has_magic(path):
            if await self._exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: await self._info(path, **kwargs)}
            elif not detail:
                return []
            else:
                return {}
        elif '/' in path[:min_idx]:
            min_idx = path[:min_idx].rindex('/')
            root = path[:min_idx + 1]
            depth = path[min_idx + 1:].count('/') + 1
        else:
            root = ''
            depth = path[min_idx + 1:].count('/') + 1
        if '**' in path:
            if maxdepth is not None:
                idx_double_stars = path.find('**')
                depth_double_stars = path[idx_double_stars:].count('/') + 1
                depth = depth - depth_double_stars + maxdepth
            else:
                depth = None
        allpaths = await self._find(root, maxdepth=depth, withdirs=True, detail=True, **kwargs)
        pattern = glob_translate(path + ('/' if ends_with_sep else ''))
        pattern = re.compile(pattern)
        out = {p: info for p, info in sorted(allpaths.items()) if pattern.match(p + '/' if append_slash_to_dirname and info['type'] == 'directory' else p)}
        if detail:
            return out
        else:
            return list(out)

    async def _du(self, path, total=True, maxdepth=None, **kwargs):
        sizes = {}
        for f in await self._find(path, maxdepth=maxdepth, **kwargs):
            info = await self._info(f)
            sizes[info['name']] = info['size']
        if total:
            return sum(sizes.values())
        else:
            return sizes

    async def _find(self, path, maxdepth=None, withdirs=False, **kwargs):
        path = self._strip_protocol(path)
        out = {}
        detail = kwargs.pop('detail', False)
        if withdirs and path != '' and await self._isdir(path):
            out[path] = await self._info(path)
        async for _, dirs, files in self._walk(path, maxdepth, detail=True, **kwargs):
            if withdirs:
                files.update(dirs)
            out.update({info['name']: info for name, info in files.items()})
        if not out and await self._isfile(path):
            out[path] = {}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    async def _expand_path(self, path, recursive=False, maxdepth=None):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        if isinstance(path, str):
            out = await self._expand_path([path], recursive, maxdepth)
        else:
            out = set()
            path = [self._strip_protocol(p) for p in path]
            for p in path:
                if has_magic(p):
                    bit = set(await self._glob(p, maxdepth=maxdepth))
                    out |= bit
                    if recursive:
                        if maxdepth is not None and maxdepth <= 1:
                            continue
                        out |= set(await self._expand_path(list(bit), recursive=recursive, maxdepth=maxdepth - 1 if maxdepth is not None else None))
                    continue
                elif recursive:
                    rec = set(await self._find(p, maxdepth=maxdepth, withdirs=True))
                    out |= rec
                if p not in out and (recursive is False or await self._exists(p)):
                    out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return sorted(out)

    async def _mkdir(self, path, create_parents=True, **kwargs):
        pass

    async def _makedirs(self, path, exist_ok=False):
        pass

    async def open_async(self, path, mode='rb', **kwargs):
        if 'b' not in mode or kwargs.get('compression'):
            raise ValueError
        raise NotImplementedError