import copy
from typing import AsyncIterable, Awaitable, Sequence
from scrapy.http import HtmlResponse, Request, Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Spider
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.spider import iterate_spider_output

This modules implements the CrawlSpider which is the recommended spider to use
for scraping typical web sites that requires crawling pages.

See documentation in docs/topics/spiders.rst
