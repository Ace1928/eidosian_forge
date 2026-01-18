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
def _get_uri_params(self, spider: Spider, uri_params_function: Optional[Union[str, Callable[[dict, Spider], dict]]], slot: Optional[FeedSlot]=None) -> dict:
    params = {}
    for k in dir(spider):
        params[k] = getattr(spider, k)
    utc_now = datetime.now(tz=timezone.utc)
    params['time'] = utc_now.replace(microsecond=0).isoformat().replace(':', '-')
    params['batch_time'] = utc_now.isoformat().replace(':', '-')
    params['batch_id'] = slot.batch_id + 1 if slot is not None else 1
    uripar_function = load_object(uri_params_function) if uri_params_function else lambda params, _: params
    new_params = uripar_function(params, spider)
    return new_params if new_params is not None else params