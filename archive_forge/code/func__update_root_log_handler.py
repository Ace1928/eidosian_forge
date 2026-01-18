from __future__ import annotations
import logging
import pprint
import signal
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Set, Type, Union, cast
from twisted.internet.defer import (
from zope.interface.exceptions import DoesNotImplement
from zope.interface.verify import verifyClass
from scrapy import Spider, signals
from scrapy.addons import AddonManager
from scrapy.core.engine import ExecutionEngine
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.extension import ExtensionManager
from scrapy.interfaces import ISpiderLoader
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings, overridden_settings
from scrapy.signalmanager import SignalManager
from scrapy.statscollectors import StatsCollector
from scrapy.utils.log import (
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.ossignal import install_shutdown_handlers, signal_names
from scrapy.utils.reactor import (
def _update_root_log_handler(self) -> None:
    if get_scrapy_root_handler() is not None:
        install_scrapy_root_handler(self.settings)