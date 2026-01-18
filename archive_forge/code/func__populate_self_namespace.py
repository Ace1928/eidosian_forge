import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _populate_self_namespace(context, template, self_ns=None):
    if self_ns is None:
        self_ns = TemplateNamespace('self:%s' % template.uri, context, template=template, populate_self=False)
    context._data['self'] = context._data['local'] = self_ns
    if hasattr(template.module, '_mako_inherit'):
        ret = template.module._mako_inherit(template, context)
        if ret:
            return ret
    return (template.callable_, context)