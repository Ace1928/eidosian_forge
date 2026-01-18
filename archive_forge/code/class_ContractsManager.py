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
class ContractsManager:
    contracts: Dict[str, Contract] = {}

    def __init__(self, contracts):
        for contract in contracts:
            self.contracts[contract.name] = contract

    def tested_methods_from_spidercls(self, spidercls):
        is_method = re.compile('^\\s*@', re.MULTILINE).search
        methods = []
        for key, value in getmembers(spidercls):
            if callable(value) and value.__doc__ and is_method(value.__doc__):
                methods.append(key)
        return methods

    def extract_contracts(self, method):
        contracts = []
        for line in method.__doc__.split('\n'):
            line = line.strip()
            if line.startswith('@'):
                name, args = re.match('@(\\w+)\\s*(.*)', line).groups()
                args = re.split('\\s+', args)
                contracts.append(self.contracts[name](method, *args))
        return contracts

    def from_spider(self, spider, results):
        requests = []
        for method in self.tested_methods_from_spidercls(type(spider)):
            bound_method = spider.__getattribute__(method)
            try:
                requests.append(self.from_method(bound_method, results))
            except Exception:
                case = _create_testcase(bound_method, 'contract')
                results.addError(case, sys.exc_info())
        return requests

    def from_method(self, method, results):
        contracts = self.extract_contracts(method)
        if contracts:
            request_cls = Request
            for contract in contracts:
                if contract.request_cls is not None:
                    request_cls = contract.request_cls
            args, kwargs = get_spec(request_cls.__init__)
            kwargs['dont_filter'] = True
            kwargs['callback'] = method
            for contract in contracts:
                kwargs = contract.adjust_request_args(kwargs)
            args.remove('self')
            if set(args).issubset(set(kwargs)):
                request = request_cls(**kwargs)
                for contract in reversed(contracts):
                    request = contract.add_pre_hook(request, results)
                for contract in contracts:
                    request = contract.add_post_hook(request, results)
                self._clean_req(request, method, results)
                return request

    def _clean_req(self, request, method, results):
        """stop the request from returning objects and records any errors"""
        cb = request.callback

        @wraps(cb)
        def cb_wrapper(response, **cb_kwargs):
            try:
                output = cb(response, **cb_kwargs)
                output = list(iterate_spider_output(output))
            except Exception:
                case = _create_testcase(method, 'callback')
                results.addError(case, sys.exc_info())

        def eb_wrapper(failure):
            case = _create_testcase(method, 'errback')
            exc_info = (failure.type, failure.value, failure.getTracebackObject())
            results.addError(case, exc_info)
        request.callback = cb_wrapper
        request.errback = eb_wrapper