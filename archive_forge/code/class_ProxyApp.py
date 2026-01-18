import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
class ProxyApp(object):
    __slots__ = ['_obj']

    def __init__(self):
        object.__init__(self)
        object.__setattr__(self, '_obj', None)

    def _ensure_app(self):
        app = object.__getattribute__(self, '_obj')
        if app is None:
            from kivy.app import App
            app = App.get_running_app()
            object.__setattr__(self, '_obj', app)
            app.bind(on_stop=lambda instance: object.__setattr__(self, '_obj', None))
        return app

    def __getattribute__(self, name):
        object.__getattribute__(self, '_ensure_app')()
        return getattr(object.__getattribute__(self, '_obj'), name)

    def __delattr__(self, name):
        object.__getattribute__(self, '_ensure_app')()
        delattr(object.__getattribute__(self, '_obj'), name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, '_ensure_app')()
        setattr(object.__getattribute__(self, '_obj'), name, value)

    def __bool__(self):
        object.__getattribute__(self, '_ensure_app')()
        return bool(object.__getattribute__(self, '_obj'))

    def __str__(self):
        object.__getattribute__(self, '_ensure_app')()
        return str(object.__getattribute__(self, '_obj'))

    def __repr__(self):
        object.__getattribute__(self, '_ensure_app')()
        return repr(object.__getattribute__(self, '_obj'))