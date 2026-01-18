from __future__ import annotations
import traceback
import warnings
from collections import defaultdict
from types import ModuleType
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Tuple, Type
from zope.interface import implementer
from scrapy import Request, Spider
from scrapy.interfaces import ISpiderLoader
from scrapy.settings import BaseSettings
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes
def _load_all_spiders(self) -> None:
    for name in self.spider_modules:
        try:
            for module in walk_modules(name):
                self._load_spiders(module)
        except ImportError:
            if self.warn_only:
                warnings.warn(f"\n{traceback.format_exc()}Could not load spiders from module '{name}'. See above traceback for details.", category=RuntimeWarning)
            else:
                raise
    self._check_name_duplicates()