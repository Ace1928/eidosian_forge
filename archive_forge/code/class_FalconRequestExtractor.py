from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class FalconRequestExtractor(RequestExtractor):

    def env(self):
        return self.request.env

    def cookies(self):
        return self.request.cookies

    def form(self):
        return None

    def files(self):
        return None

    def raw_data(self):
        content_length = self.content_length()
        if content_length > 0:
            return '[REQUEST_CONTAINING_RAW_DATA]'
        else:
            return None
    if FALCON3:

        def json(self):
            try:
                return self.request.media
            except falcon.errors.HTTPBadRequest:
                return None
    else:

        def json(self):
            try:
                return self.request.media
            except falcon.errors.HTTPBadRequest:
                return self.request._media