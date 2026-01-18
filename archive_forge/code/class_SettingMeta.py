import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class SettingMeta(type):

    def __new__(cls, name, bases, attrs):
        super_new = super().__new__
        parents = [b for b in bases if isinstance(b, SettingMeta)]
        if not parents:
            return super_new(cls, name, bases, attrs)
        attrs['order'] = len(KNOWN_SETTINGS)
        attrs['validator'] = staticmethod(attrs['validator'])
        new_class = super_new(cls, name, bases, attrs)
        new_class.fmt_desc(attrs.get('desc', ''))
        KNOWN_SETTINGS.append(new_class)
        return new_class

    def fmt_desc(cls, desc):
        desc = textwrap.dedent(desc).strip()
        setattr(cls, 'desc', desc)
        setattr(cls, 'short', desc.splitlines()[0])