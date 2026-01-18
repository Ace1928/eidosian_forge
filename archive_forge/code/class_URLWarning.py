import logging
import re
import warnings
from scrapy import signals
from scrapy.http import Request
from scrapy.utils.httpobj import urlparse_cached
class URLWarning(Warning):
    pass