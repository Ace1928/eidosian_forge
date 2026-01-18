import collections
import logging
import os
import re
import sys
import time
import warnings
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from itertools import chain
from types import SimpleNamespace
from unittest import TestCase, skipIf, skipUnless
from xml.dom.minidom import Node, parseString
from asgiref.sync import iscoroutinefunction
from django.apps import apps
from django.apps.registry import Apps
from django.conf import UserSettingsHolder, settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import request_started, setting_changed
from django.db import DEFAULT_DB_ALIAS, connections, reset_queries
from django.db.models.options import Options
from django.template import Template
from django.test.signals import template_rendered
from django.urls import get_script_prefix, set_script_prefix
from django.utils.translation import deactivate
class override_settings(TestContextDecorator):
    """
    Act as either a decorator or a context manager. If it's a decorator, take a
    function and return a wrapped function. If it's a contextmanager, use it
    with the ``with`` statement. In either event, entering/exiting are called
    before and after, respectively, the function/block is executed.
    """
    enable_exception = None

    def __init__(self, **kwargs):
        self.options = kwargs
        super().__init__()

    def enable(self):
        if 'INSTALLED_APPS' in self.options:
            try:
                apps.set_installed_apps(self.options['INSTALLED_APPS'])
            except Exception:
                apps.unset_installed_apps()
                raise
        override = UserSettingsHolder(settings._wrapped)
        for key, new_value in self.options.items():
            setattr(override, key, new_value)
        self.wrapped = settings._wrapped
        settings._wrapped = override
        for key, new_value in self.options.items():
            try:
                setting_changed.send(sender=settings._wrapped.__class__, setting=key, value=new_value, enter=True)
            except Exception as exc:
                self.enable_exception = exc
                self.disable()

    def disable(self):
        if 'INSTALLED_APPS' in self.options:
            apps.unset_installed_apps()
        settings._wrapped = self.wrapped
        del self.wrapped
        responses = []
        for key in self.options:
            new_value = getattr(settings, key, None)
            responses_for_setting = setting_changed.send_robust(sender=settings._wrapped.__class__, setting=key, value=new_value, enter=False)
            responses.extend(responses_for_setting)
        if self.enable_exception is not None:
            exc = self.enable_exception
            self.enable_exception = None
            raise exc
        for _, response in responses:
            if isinstance(response, Exception):
                raise response

    def save_options(self, test_func):
        if test_func._overridden_settings is None:
            test_func._overridden_settings = self.options
        else:
            test_func._overridden_settings = {**test_func._overridden_settings, **self.options}

    def decorate_class(self, cls):
        from django.test import SimpleTestCase
        if not issubclass(cls, SimpleTestCase):
            raise ValueError('Only subclasses of Django SimpleTestCase can be decorated with override_settings')
        self.save_options(cls)
        return cls