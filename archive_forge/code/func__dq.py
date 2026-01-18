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
def _dq(self):
    """Create a new priority queue instance, with disk storage"""
    assert self.dqdir
    state = self._read_dqs_state(self.dqdir)
    q = create_instance(self.pqclass, settings=None, crawler=self.crawler, downstream_queue_cls=self.dqclass, key=self.dqdir, startprios=state)
    if q:
        logger.info('Resuming crawl (%(queuesize)d requests scheduled)', {'queuesize': len(q)}, extra={'spider': self.spider})
    return q