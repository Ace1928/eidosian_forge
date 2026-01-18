import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse
from twisted.internet import defer, threads
from twisted.internet.defer import DeferredList
from w3lib.url import file_uri_to_path
from zope.interface import Interface, implementer
from scrapy import Spider, signals
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.extensions.postprocessing import PostProcessingManager
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.conf import feed_complete_default_values_from_settings
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import get_func_args, without_none_values
def _settings_are_valid(self):
    """
        If FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified uri has to contain
        %(batch_time)s or %(batch_id)d to distinguish different files of partial output
        """
    for uri_template, values in self.feeds.items():
        if values['batch_item_count'] and (not re.search('%\\(batch_time\\)s|%\\(batch_id\\)', uri_template)):
            logger.error('%%(batch_time)s or %%(batch_id)d must be in the feed URI (%s) if FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified and greater than 0. For more info see: https://docs.scrapy.org/en/latest/topics/feed-exports.html#feed-export-batch-item-count', uri_template)
            return False
    return True