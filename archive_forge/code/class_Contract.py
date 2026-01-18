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
class Contract:
    """Abstract class for contracts"""
    request_cls = None

    def __init__(self, method, *args):
        self.testcase_pre = _create_testcase(method, f'@{self.name} pre-hook')
        self.testcase_post = _create_testcase(method, f'@{self.name} post-hook')
        self.args = args

    def add_pre_hook(self, request, results):
        if hasattr(self, 'pre_process'):
            cb = request.callback

            @wraps(cb)
            def wrapper(response, **cb_kwargs):
                try:
                    results.startTest(self.testcase_pre)
                    self.pre_process(response)
                    results.stopTest(self.testcase_pre)
                except AssertionError:
                    results.addFailure(self.testcase_pre, sys.exc_info())
                except Exception:
                    results.addError(self.testcase_pre, sys.exc_info())
                else:
                    results.addSuccess(self.testcase_pre)
                finally:
                    cb_result = cb(response, **cb_kwargs)
                    if isinstance(cb_result, (AsyncGenerator, CoroutineType)):
                        raise TypeError("Contracts don't support async callbacks")
                    return list(iterate_spider_output(cb_result))
            request.callback = wrapper
        return request

    def add_post_hook(self, request, results):
        if hasattr(self, 'post_process'):
            cb = request.callback

            @wraps(cb)
            def wrapper(response, **cb_kwargs):
                cb_result = cb(response, **cb_kwargs)
                if isinstance(cb_result, (AsyncGenerator, CoroutineType)):
                    raise TypeError("Contracts don't support async callbacks")
                output = list(iterate_spider_output(cb_result))
                try:
                    results.startTest(self.testcase_post)
                    self.post_process(output)
                    results.stopTest(self.testcase_post)
                except AssertionError:
                    results.addFailure(self.testcase_post, sys.exc_info())
                except Exception:
                    results.addError(self.testcase_post, sys.exc_info())
                else:
                    results.addSuccess(self.testcase_post)
                finally:
                    return output
            request.callback = wrapper
        return request

    def adjust_request_args(self, args):
        return args