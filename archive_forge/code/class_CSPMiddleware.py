from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
class CSPMiddleware(MiddlewareMixin):
    """The admin's JavaScript should be compatible with CSP."""

    def process_response(self, request, response):
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response