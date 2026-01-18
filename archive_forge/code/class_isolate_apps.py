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
class isolate_apps(TestContextDecorator):
    """
    Act as either a decorator or a context manager to register models defined
    in its wrapped context to an isolated registry.

    The list of installed apps the isolated registry should contain must be
    passed as arguments.

    Two optional keyword arguments can be specified:

    `attr_name`: attribute assigned the isolated registry if used as a class
                 decorator.

    `kwarg_name`: keyword argument passing the isolated registry if used as a
                  function decorator.
    """

    def __init__(self, *installed_apps, **kwargs):
        self.installed_apps = installed_apps
        super().__init__(**kwargs)

    def enable(self):
        self.old_apps = Options.default_apps
        apps = Apps(self.installed_apps)
        setattr(Options, 'default_apps', apps)
        return apps

    def disable(self):
        setattr(Options, 'default_apps', self.old_apps)