from __future__ import annotations
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, cast
from twisted.internet.defer import Deferred
from scrapy.crawler import Crawler
from scrapy.dupefilters import BaseDupeFilter
from scrapy.http.request import Request
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.job import job_dir
from scrapy.utils.misc import create_instance, load_object
def _dqdir(self, jobdir: Optional[str]) -> Optional[str]:
    """Return a folder name to keep disk queue state at"""
    if jobdir is not None:
        dqdir = Path(jobdir, 'requests.queue')
        if not dqdir.exists():
            dqdir.mkdir(parents=True)
        return str(dqdir)
    return None