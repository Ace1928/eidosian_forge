import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
class _ApplicationRouter(ReversibleRuleRouter):
    """Routing implementation used internally by `Application`.

    Provides a binding between `Application` and `RequestHandler`.
    This implementation extends `~.routing.ReversibleRuleRouter` in a couple of ways:
        * it allows to use `RequestHandler` subclasses as `~.routing.Rule` target and
        * it allows to use a list/tuple of rules as `~.routing.Rule` target.
        ``process_rule`` implementation will substitute this list with an appropriate
        `_ApplicationRouter` instance.
    """

    def __init__(self, application: 'Application', rules: Optional[_RuleList]=None) -> None:
        assert isinstance(application, Application)
        self.application = application
        super().__init__(rules)

    def process_rule(self, rule: Rule) -> Rule:
        rule = super().process_rule(rule)
        if isinstance(rule.target, (list, tuple)):
            rule.target = _ApplicationRouter(self.application, rule.target)
        return rule

    def get_target_delegate(self, target: Any, request: httputil.HTTPServerRequest, **target_params: Any) -> Optional[httputil.HTTPMessageDelegate]:
        if isclass(target) and issubclass(target, RequestHandler):
            return self.application.get_handler_delegate(request, target, **target_params)
        return super().get_target_delegate(target, request, **target_params)