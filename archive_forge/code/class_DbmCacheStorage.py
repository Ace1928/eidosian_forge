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
class DbmCacheStorage:

    def __init__(self, settings):
        self.cachedir = data_path(settings['HTTPCACHE_DIR'], createdir=True)
        self.expiration_secs = settings.getint('HTTPCACHE_EXPIRATION_SECS')
        self.dbmodule = import_module(settings['HTTPCACHE_DBM_MODULE'])
        self.db = None

    def open_spider(self, spider: Spider):
        dbpath = Path(self.cachedir, f'{spider.name}.db')
        self.db = self.dbmodule.open(str(dbpath), 'c')
        logger.debug('Using DBM cache storage in %(cachepath)s', {'cachepath': dbpath}, extra={'spider': spider})
        self._fingerprinter = spider.crawler.request_fingerprinter

    def close_spider(self, spider):
        self.db.close()

    def retrieve_response(self, spider, request):
        data = self._read_data(spider, request)
        if data is None:
            return
        url = data['url']
        status = data['status']
        headers = Headers(data['headers'])
        body = data['body']
        respcls = responsetypes.from_args(headers=headers, url=url, body=body)
        response = respcls(url=url, headers=headers, status=status, body=body)
        return response

    def store_response(self, spider, request, response):
        key = self._fingerprinter.fingerprint(request).hex()
        data = {'status': response.status, 'url': response.url, 'headers': dict(response.headers), 'body': response.body}
        self.db[f'{key}_data'] = pickle.dumps(data, protocol=4)
        self.db[f'{key}_time'] = str(time())

    def _read_data(self, spider, request):
        key = self._fingerprinter.fingerprint(request).hex()
        db = self.db
        tkey = f'{key}_time'
        if tkey not in db:
            return
        ts = db[tkey]
        if 0 < self.expiration_secs < time() - float(ts):
            return
        return pickle.loads(db[f'{key}_data'])