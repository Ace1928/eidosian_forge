from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def add_api_log_filters(modules: typing.Optional[typing.Union[typing.List[str], str]]=['gunicorn', 'uvicorn'], routes: typing.Optional[typing.Union[typing.List[str], str]]=['/healthz'], status_codes: typing.Optional[typing.Union[typing.List[int], int]]=None, verbose: bool=False):
    """
    Add filters to the logger to remove health checks and other unwanted logs

    args:

        modules: list of modules to filter [default: ['gunicorn', 'uvicorn']
        routes: list of routes to filter [default: ['/healthz']]
        status_codes: list of status codes to filter [default: None]
        verbose: bool = False [default: False]
    """
    if not isinstance(modules, list):
        modules = [modules]
    if routes and (not isinstance(routes, list)):
        routes = [routes]
    if status_codes and (not isinstance(status_codes, list)):
        status_codes = [status_codes]

    def filter_api_record(record: logging.LogRecord) -> bool:
        """
        Filter out health checks and other unwanted logs
        """
        if routes:
            for route in routes:
                if route in record.args:
                    return False
        if status_codes:
            for sc in status_codes:
                if sc in record.args:
                    return False
        return True
    for module in modules:
        if module == 'gunicorn':
            module = 'gunicorn.glogging.Logger'
        elif module == 'uvicorn':
            module = 'uvicorn.logging.Logger'
        _apilogger = logging.getLogger(module)
        if verbose:
            default_logger.info(f'Adding API filters to {module} for routes: {routes} and status_codes: {status_codes}')
        _apilogger.addFilter(filter_api_record)