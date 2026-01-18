import asyncio
import io
import logging
import re
import weakref
from copy import copy
from urllib.parse import urlparse
import aiohttp
import yarl
from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem, sync, sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.exceptions import FSTimeoutError
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import (
from ..caching import AllBytes
class HTTPFileSystem(AsyncFileSystem):
    """
    Simple File-System for fetching data via HTTP(S)

    ``ls()`` is implemented by loading the parent page and doing a regex
    match on the result. If simple_link=True, anything of the form
    "http(s)://server.com/stuff?thing=other"; otherwise only links within
    HTML href tags will be used.
    """
    sep = '/'

    def __init__(self, simple_links=True, block_size=None, same_scheme=True, size_policy=None, cache_type='bytes', cache_options=None, asynchronous=False, loop=None, client_kwargs=None, get_client=get_client, encoded=False, **storage_options):
        """
        NB: if this is called async, you must await set_client

        Parameters
        ----------
        block_size: int
            Blocks to read bytes; if 0, will default to raw requests file-like
            objects instead of HTTPFile instances
        simple_links: bool
            If True, will consider both HTML <a> tags and anything that looks
            like a URL; if False, will consider only the former.
        same_scheme: True
            When doing ls/glob, if this is True, only consider paths that have
            http/https matching the input URLs.
        size_policy: this argument is deprecated
        client_kwargs: dict
            Passed to aiohttp.ClientSession, see
            https://docs.aiohttp.org/en/stable/client_reference.html
            For example, ``{'auth': aiohttp.BasicAuth('user', 'pass')}``
        get_client: Callable[..., aiohttp.ClientSession]
            A callable which takes keyword arguments and constructs
            an aiohttp.ClientSession. It's state will be managed by
            the HTTPFileSystem class.
        storage_options: key-value
            Any other parameters passed on to requests
        cache_type, cache_options: defaults used in open
        """
        super().__init__(self, asynchronous=asynchronous, loop=loop, **storage_options)
        self.block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        self.simple_links = simple_links
        self.same_schema = same_scheme
        self.cache_type = cache_type
        self.cache_options = cache_options
        self.client_kwargs = client_kwargs or {}
        self.get_client = get_client
        self.encoded = encoded
        self.kwargs = storage_options
        self._session = None
        request_options = copy(storage_options)
        self.use_listings_cache = request_options.pop('use_listings_cache', False)
        request_options.pop('listings_expiry_time', None)
        request_options.pop('max_paths', None)
        request_options.pop('skip_instance_cache', None)
        self.kwargs = request_options

    @property
    def fsid(self):
        return 'http'

    def encode_url(self, url):
        return yarl.URL(url, encoded=self.encoded)

    @staticmethod
    def close_session(loop, session):
        if loop is not None and loop.is_running():
            try:
                sync(loop, session.close, timeout=0.1)
                return
            except (TimeoutError, FSTimeoutError, NotImplementedError):
                pass
        connector = getattr(session, '_connector', None)
        if connector is not None:
            connector._close()

    async def set_session(self):
        if self._session is None:
            self._session = await self.get_client(loop=self.loop, **self.client_kwargs)
            if not self.asynchronous:
                weakref.finalize(self, self.close_session, self.loop, self._session)
        return self._session

    @classmethod
    def _strip_protocol(cls, path):
        """For HTTP, we always want to keep the full URL"""
        return path

    @classmethod
    def _parent(cls, path):
        par = super()._parent(path)
        if len(par) > 7:
            return par
        return ''

    async def _ls_real(self, url, detail=True, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)
        session = await self.set_session()
        async with session.get(self.encode_url(url), **self.kwargs) as r:
            self._raise_not_found_for_status(r, url)
            try:
                text = await r.text()
                if self.simple_links:
                    links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
                else:
                    links = [u[2] for u in ex.findall(text)]
            except UnicodeDecodeError:
                links = []
        out = set()
        parts = urlparse(url)
        for l in links:
            if isinstance(l, tuple):
                l = l[1]
            if l.startswith('/') and len(l) > 1:
                l = f'{parts.scheme}://{parts.netloc}{l}'
            if l.startswith('http'):
                if self.same_schema and l.startswith(url.rstrip('/') + '/'):
                    out.add(l)
                elif l.replace('https', 'http').startswith(url.replace('https', 'http').rstrip('/') + '/'):
                    out.add(l)
            elif l not in ['..', '../']:
                out.add('/'.join([url.rstrip('/'), l.lstrip('/')]))
        if not out and url.endswith('/'):
            out = await self._ls_real(url.rstrip('/'), detail=False)
        if detail:
            return [{'name': u, 'size': None, 'type': 'directory' if u.endswith('/') else 'file'} for u in out]
        else:
            return sorted(out)

    async def _ls(self, url, detail=True, **kwargs):
        if self.use_listings_cache and url in self.dircache:
            out = self.dircache[url]
        else:
            out = await self._ls_real(url, detail=detail, **kwargs)
            self.dircache[url] = out
        return out
    ls = sync_wrapper(_ls)

    def _raise_not_found_for_status(self, response, url):
        """
        Raises FileNotFoundError for 404s, otherwise uses raise_for_status.
        """
        if response.status == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()

    async def _cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)
        if start is not None or end is not None:
            if start == end:
                return b''
            headers = kw.pop('headers', {}).copy()
            headers['Range'] = await self._process_limits(url, start, end)
            kw['headers'] = headers
        session = await self.set_session()
        async with session.get(self.encode_url(url), **kw) as r:
            out = await r.read()
            self._raise_not_found_for_status(r, url)
        return out

    async def _get_file(self, rpath, lpath, chunk_size=5 * 2 ** 20, callback=DEFAULT_CALLBACK, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(rpath)
        session = await self.set_session()
        async with session.get(self.encode_url(rpath), **kw) as r:
            try:
                size = int(r.headers['content-length'])
            except (ValueError, KeyError):
                size = None
            callback.set_size(size)
            self._raise_not_found_for_status(r, rpath)
            if isfilelike(lpath):
                outfile = lpath
            else:
                outfile = open(lpath, 'wb')
            try:
                chunk = True
                while chunk:
                    chunk = await r.content.read(chunk_size)
                    outfile.write(chunk)
                    callback.relative_update(len(chunk))
            finally:
                if not isfilelike(lpath):
                    outfile.close()

    async def _put_file(self, lpath, rpath, chunk_size=5 * 2 ** 20, callback=DEFAULT_CALLBACK, method='post', **kwargs):

        async def gen_chunks():
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False
            else:
                context = open(lpath, 'rb')
                use_seek = True
            with context as f:
                if use_seek:
                    callback.set_size(f.seek(0, 2))
                    f.seek(0)
                else:
                    callback.set_size(getattr(f, 'size', None))
                chunk = f.read(chunk_size)
                while chunk:
                    yield chunk
                    callback.relative_update(len(chunk))
                    chunk = f.read(chunk_size)
        kw = self.kwargs.copy()
        kw.update(kwargs)
        session = await self.set_session()
        method = method.lower()
        if method not in ('post', 'put'):
            raise ValueError(f"method has to be either 'post' or 'put', not: {method!r}")
        meth = getattr(session, method)
        async with meth(self.encode_url(rpath), data=gen_chunks(), **kw) as resp:
            self._raise_not_found_for_status(resp, rpath)

    async def _exists(self, path, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        try:
            logger.debug(path)
            session = await self.set_session()
            r = await session.get(self.encode_url(path), **kw)
            async with r:
                return r.status < 400
        except aiohttp.ClientError:
            return False

    async def _isfile(self, path, **kwargs):
        return await self._exists(path, **kwargs)

    def _open(self, path, mode='rb', block_size=None, autocommit=None, cache_type=None, cache_options=None, size=None, **kwargs):
        """Make a file-like object

        Parameters
        ----------
        path: str
            Full URL with protocol
        mode: string
            must be "rb"
        block_size: int or None
            Bytes to download in one request; use instance value if None. If
            zero, will return a streaming Requests file-like instance.
        kwargs: key-value
            Any other parameters, passed to requests calls
        """
        if mode != 'rb':
            raise NotImplementedError
        block_size = block_size if block_size is not None else self.block_size
        kw = self.kwargs.copy()
        kw['asynchronous'] = self.asynchronous
        kw.update(kwargs)
        size = size or self.info(path, **kwargs)['size']
        session = sync(self.loop, self.set_session)
        if block_size and size:
            return HTTPFile(self, path, session=session, block_size=block_size, mode=mode, size=size, cache_type=cache_type or self.cache_type, cache_options=cache_options or self.cache_options, loop=self.loop, **kw)
        else:
            return HTTPStreamFile(self, path, mode=mode, loop=self.loop, session=session, **kw)

    async def open_async(self, path, mode='rb', size=None, **kwargs):
        session = await self.set_session()
        if size is None:
            try:
                size = (await self._info(path, **kwargs))['size']
            except FileNotFoundError:
                pass
        return AsyncStreamFile(self, path, loop=self.loop, session=session, size=size, **kwargs)

    def ukey(self, url):
        """Unique identifier; assume HTTP files are static, unchanging"""
        return tokenize(url, self.kwargs, self.protocol)

    async def _info(self, url, **kwargs):
        """Get info of URL

        Tries to access location via HEAD, and then GET methods, but does
        not fetch the data.

        It is possible that the server does not supply any size information, in
        which case size will be given as None (and certain operations on the
        corresponding file will not work).
        """
        info = {}
        session = await self.set_session()
        for policy in ['head', 'get']:
            try:
                info.update(await _file_info(self.encode_url(url), size_policy=policy, session=session, **self.kwargs, **kwargs))
                if info.get('size') is not None:
                    break
            except Exception as exc:
                if policy == 'get':
                    raise FileNotFoundError(url) from exc
                logger.debug('', exc_info=exc)
        return {'name': url, 'size': None, **info, 'type': 'file'}

    async def _glob(self, path, maxdepth=None, **kwargs):
        """
        Find files by glob-matching.

        This implementation is idntical to the one in AbstractFileSystem,
        but "?" is not considered as a character for globbing, because it is
        so common in URLs, often identifying the "query" part.
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        import re
        ends_with_slash = path.endswith('/')
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_slash or path.endswith('/**')
        idx_star = path.find('*') if path.find('*') >= 0 else len(path)
        idx_brace = path.find('[') if path.find('[') >= 0 else len(path)
        min_idx = min(idx_star, idx_brace)
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
        pattern = glob_translate(path + ('/' if ends_with_slash else ''))
        pattern = re.compile(pattern)
        out = {p: info for p, info in sorted(allpaths.items()) if pattern.match(p + '/' if append_slash_to_dirname and info['type'] == 'directory' else p)}
        if detail:
            return out
        else:
            return list(out)

    async def _isdir(self, path):
        try:
            return bool(await self._ls(path))
        except (FileNotFoundError, ValueError):
            return False