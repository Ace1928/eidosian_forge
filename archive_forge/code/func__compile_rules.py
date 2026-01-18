import copy
from typing import AsyncIterable, Awaitable, Sequence
from scrapy.http import HtmlResponse, Request, Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Spider
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.spider import iterate_spider_output
def _compile_rules(self):
    self._rules = []
    for rule in self.rules:
        self._rules.append(copy.copy(rule))
        self._rules[-1]._compile(self)