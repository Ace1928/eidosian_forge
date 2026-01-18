from __future__ import absolute_import
import os
import sys
import weakref
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._compat import reraise, iteritems
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk._types import TYPE_CHECKING
class PyramidRequestExtractor(RequestExtractor):

    def url(self):
        return self.request.path_url

    def env(self):
        return self.request.environ

    def cookies(self):
        return self.request.cookies

    def raw_data(self):
        return self.request.text

    def form(self):
        return {key: value for key, value in iteritems(self.request.POST) if not getattr(value, 'filename', None)}

    def files(self):
        return {key: value for key, value in iteritems(self.request.POST) if getattr(value, 'filename', None)}

    def size_of_file(self, postdata):
        file = postdata.file
        try:
            return os.fstat(file.fileno()).st_size
        except Exception:
            return 0