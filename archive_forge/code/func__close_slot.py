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
def _close_slot(self, slot, spider):

    def get_file(slot_):
        if isinstance(slot_.file, PostProcessingManager):
            slot_.file.close()
            return slot_.file.file
        return slot_.file
    if slot.itemcount:
        slot.finish_exporting()
    elif slot.store_empty and slot.batch_id == 1:
        slot.start_exporting()
        slot.finish_exporting()
    else:
        return None
    logmsg = f'{slot.format} feed ({slot.itemcount} items) in: {slot.uri}'
    d = defer.maybeDeferred(slot.storage.store, get_file(slot))
    d.addCallback(self._handle_store_success, logmsg, spider, type(slot.storage).__name__)
    d.addErrback(self._handle_store_error, logmsg, spider, type(slot.storage).__name__)
    self._pending_deferreds.append(d)
    d.addCallback(lambda _: self.crawler.signals.send_catch_log_deferred(signals.feed_slot_closed, slot=slot))
    d.addBoth(lambda _: self._pending_deferreds.remove(d))
    return d