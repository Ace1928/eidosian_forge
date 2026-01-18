from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _interpret_inherit(self, body, defs, inherit_template, ns):
    __traceback_hide__ = True
    if not self.get_template:
        raise TemplateError('You cannot use inheritance without passing in get_template', position=None, name=self.name)
    templ = self.get_template(inherit_template, self)
    self_ = TemplateObject(self.name)
    for name, value in defs.items():
        setattr(self_, name, value)
    self_.body = body
    ns = ns.copy()
    ns['self'] = self_
    return templ.substitute(ns)