import logging
import re
from typing import TYPE_CHECKING, Any
from scrapy.http import Request, XmlResponse
from scrapy.spiders import Spider
from scrapy.utils._compression import _DecompressionMaxSizeExceeded
from scrapy.utils.gz import gunzip, gzip_magic_number
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
def _get_sitemap_body(self, response):
    """Return the sitemap body contained in the given response,
        or None if the response is not a sitemap.
        """
    if isinstance(response, XmlResponse):
        return response.body
    if gzip_magic_number(response):
        uncompressed_size = len(response.body)
        max_size = response.meta.get('download_maxsize', self._max_size)
        warn_size = response.meta.get('download_warnsize', self._warn_size)
        try:
            body = gunzip(response.body, max_size=max_size)
        except _DecompressionMaxSizeExceeded:
            return None
        if uncompressed_size < warn_size <= len(body):
            logger.warning(f'{response} body size after decompression ({len(body)} B) is larger than the download warning size ({warn_size} B).')
        return body
    if response.url.endswith('.xml') or response.url.endswith('.xml.gz'):
        return response.body