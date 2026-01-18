import os
import platform as _platform
import re
from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy
from types import ModuleType
from kombu.utils.url import maybe_sanitize_url
from celery.exceptions import ImproperlyConfigured
from celery.platforms import pyimplementation
from celery.utils.collections import ConfigurationView
from celery.utils.imports import import_from_cwd, qualname, symbol_by_name
from celery.utils.text import pretty
from .defaults import _OLD_DEFAULTS, _OLD_SETTING_KEYS, _TO_NEW_KEY, _TO_OLD_KEY, DEFAULTS, SETTING_KEYS, find
class AppPickler:
    """Old application pickler/unpickler (< 3.1)."""

    def __call__(self, cls, *args):
        kwargs = self.build_kwargs(*args)
        app = self.construct(cls, **kwargs)
        self.prepare(app, **kwargs)
        return app

    def prepare(self, app, **kwargs):
        app.conf.update(kwargs['changes'])

    def build_kwargs(self, *args):
        return self.build_standard_kwargs(*args)

    def build_standard_kwargs(self, main, changes, loader, backend, amqp, events, log, control, accept_magic_kwargs, config_source=None):
        return {'main': main, 'loader': loader, 'backend': backend, 'amqp': amqp, 'changes': changes, 'events': events, 'log': log, 'control': control, 'set_as_current': False, 'config_source': config_source}

    def construct(self, cls, **kwargs):
        return cls(**kwargs)