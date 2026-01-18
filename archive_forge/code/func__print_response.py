import sys
from argparse import Namespace
from typing import List, Type
from w3lib.url import is_url
from scrapy import Spider
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.http import Request
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.spider import DefaultSpider, spidercls_for_request
def _print_response(self, response, opts):
    if opts.headers:
        self._print_headers(response.request.headers, b'>')
        print('>')
        self._print_headers(response.headers, b'<')
    else:
        self._print_bytes(response.body)