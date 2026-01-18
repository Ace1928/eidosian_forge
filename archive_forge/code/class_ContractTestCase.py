import re
import sys
from functools import wraps
from inspect import getmembers
from types import CoroutineType
from typing import AsyncGenerator, Dict
from unittest import TestCase
from scrapy.http import Request
from scrapy.utils.python import get_spec
from scrapy.utils.spider import iterate_spider_output
class ContractTestCase(TestCase):

    def __str__(_self):
        return f'[{spider}] {method.__name__} ({desc})'