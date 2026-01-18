import logging
import operator
from functools import partial
from urllib.parse import urljoin, urlparse
from lxml import etree
from parsel.csstranslator import HTMLTranslator
from w3lib.html import strip_html5_whitespace
from w3lib.url import canonicalize_url, safe_url_string
from scrapy.link import Link
from scrapy.linkextractors import (
from scrapy.utils.misc import arg_to_iter, rel_has_nofollow
from scrapy.utils.python import unique as unique_list
from scrapy.utils.response import get_base_url
from scrapy.utils.url import url_has_any_extension, url_is_from_any_domain
def _canonicalize_link_url(link):
    return canonicalize_url(link.url, keep_fragments=True)