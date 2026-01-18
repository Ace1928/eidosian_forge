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
def _ensure_app(self):
    app = object.__getattribute__(self, '_obj')
    if app is None:
        from kivy.app import App
        app = App.get_running_app()
        object.__setattr__(self, '_obj', app)
        app.bind(on_stop=lambda instance: object.__setattr__(self, '_obj', None))
    return app