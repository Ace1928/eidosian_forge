import os
import signal
from itemadapter import is_item
from twisted.internet import defer, threads
from twisted.python import threadable
from w3lib.url import any_to_uri
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.conf import get_config
from scrapy.utils.console import DEFAULT_PYTHON_SHELLS, start_python_console
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.misc import load_object
from scrapy.utils.reactor import is_asyncio_reactor_installed, set_asyncio_event_loop
from scrapy.utils.response import open_in_browser
def _is_relevant(self, value):
    return isinstance(value, self.relevant_classes) or is_item(value)