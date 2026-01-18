import sys
import weakref
from inspect import isawaitable
from sentry_sdk import continue_trace
from sentry_sdk._compat import urlparse, reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT, TRANSACTION_SOURCE_URL
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor, _filter_headers
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
class SanicRequestExtractor(RequestExtractor):

    def content_length(self):
        if self.request.body is None:
            return 0
        return len(self.request.body)

    def cookies(self):
        return dict(self.request.cookies)

    def raw_data(self):
        return self.request.body

    def form(self):
        return self.request.form

    def is_json(self):
        raise NotImplementedError()

    def json(self):
        return self.request.json

    def files(self):
        return self.request.files

    def size_of_file(self, file):
        return len(file.body or ())