import gzip
import logging
import pickle
from email.utils import mktime_tz, parsedate_tz
from importlib import import_module
from pathlib import Path
from time import time
from weakref import WeakKeyDictionary
from w3lib.http import headers_dict_to_raw, headers_raw_to_dict
from scrapy.http import Headers, Response
from scrapy.http.request import Request
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.project import data_path
from scrapy.utils.python import to_bytes, to_unicode
class FilesystemCacheStorage:

    def __init__(self, settings):
        self.cachedir = data_path(settings['HTTPCACHE_DIR'])
        self.expiration_secs = settings.getint('HTTPCACHE_EXPIRATION_SECS')
        self.use_gzip = settings.getbool('HTTPCACHE_GZIP')
        self._open = gzip.open if self.use_gzip else open

    def open_spider(self, spider: Spider):
        logger.debug('Using filesystem cache storage in %(cachedir)s', {'cachedir': self.cachedir}, extra={'spider': spider})
        assert spider.crawler.request_fingerprinter
        self._fingerprinter = spider.crawler.request_fingerprinter

    def close_spider(self, spider):
        pass

    def retrieve_response(self, spider: Spider, request: Request):
        """Return response if present in cache, or None otherwise."""
        metadata = self._read_meta(spider, request)
        if metadata is None:
            return
        rpath = Path(self._get_request_path(spider, request))
        with self._open(rpath / 'response_body', 'rb') as f:
            body = f.read()
        with self._open(rpath / 'response_headers', 'rb') as f:
            rawheaders = f.read()
        url = metadata.get('response_url')
        status = metadata['status']
        headers = Headers(headers_raw_to_dict(rawheaders))
        respcls = responsetypes.from_args(headers=headers, url=url, body=body)
        response = respcls(url=url, headers=headers, status=status, body=body)
        return response

    def store_response(self, spider: Spider, request: Request, response):
        """Store the given response in the cache."""
        rpath = Path(self._get_request_path(spider, request))
        if not rpath.exists():
            rpath.mkdir(parents=True)
        metadata = {'url': request.url, 'method': request.method, 'status': response.status, 'response_url': response.url, 'timestamp': time()}
        with self._open(rpath / 'meta', 'wb') as f:
            f.write(to_bytes(repr(metadata)))
        with self._open(rpath / 'pickled_meta', 'wb') as f:
            pickle.dump(metadata, f, protocol=4)
        with self._open(rpath / 'response_headers', 'wb') as f:
            f.write(headers_dict_to_raw(response.headers))
        with self._open(rpath / 'response_body', 'wb') as f:
            f.write(response.body)
        with self._open(rpath / 'request_headers', 'wb') as f:
            f.write(headers_dict_to_raw(request.headers))
        with self._open(rpath / 'request_body', 'wb') as f:
            f.write(request.body)

    def _get_request_path(self, spider: Spider, request: Request) -> str:
        key = self._fingerprinter.fingerprint(request).hex()
        return str(Path(self.cachedir, spider.name, key[0:2], key))

    def _read_meta(self, spider: Spider, request: Request):
        rpath = Path(self._get_request_path(spider, request))
        metapath = rpath / 'pickled_meta'
        if not metapath.exists():
            return
        mtime = metapath.stat().st_mtime
        if 0 < self.expiration_secs < time() - mtime:
            return
        with self._open(metapath, 'rb') as f:
            return pickle.load(f)