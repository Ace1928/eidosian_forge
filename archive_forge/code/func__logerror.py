import logging
from twisted.internet.defer import Deferred, maybeDeferred
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import load_object
def _logerror(self, failure, request, spider):
    if failure.type is not IgnoreRequest:
        logger.error('Error downloading %(request)s: %(f_exception)s', {'request': request, 'f_exception': failure.value}, exc_info=failure_to_exc_info(failure), extra={'spider': spider})
    return failure