import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
@tag('selenium')
class SeleniumTestCase(LiveServerTestCase, metaclass=SeleniumTestCaseBase):
    implicit_wait = 10
    external_host = None

    @classproperty
    def live_server_url(cls):
        return 'http://%s:%s' % (cls.external_host or cls.host, cls.server_thread.port)

    @classproperty
    def allowed_host(cls):
        return cls.external_host or cls.host

    @classmethod
    def setUpClass(cls):
        cls.selenium = cls.create_webdriver()
        cls.selenium.implicitly_wait(cls.implicit_wait)
        super().setUpClass()
        cls.addClassCleanup(cls._quit_selenium)

    @classmethod
    def _quit_selenium(cls):
        if hasattr(cls, 'selenium'):
            cls.selenium.quit()

    @contextmanager
    def disable_implicit_wait(self):
        """Disable the default implicit wait."""
        self.selenium.implicitly_wait(0)
        try:
            yield
        finally:
            self.selenium.implicitly_wait(self.implicit_wait)