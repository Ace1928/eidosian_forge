from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _importer(target):
    components = target.split('.')
    import_path = components.pop(0)
    thing = __import__(import_path)
    for comp in components:
        import_path += '.%s' % comp
        thing = _dot_lookup(thing, comp, import_path)
    return thing