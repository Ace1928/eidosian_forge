from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _mock_add_spec(self, spec, spec_set, _spec_as_instance=False, _eat_self=False):
    _spec_class = None
    _spec_signature = None
    if spec is not None and (not _is_list(spec)):
        if isinstance(spec, ClassTypes):
            _spec_class = spec
        else:
            _spec_class = _get_class(spec)
        res = _get_signature_object(spec, _spec_as_instance, _eat_self)
        _spec_signature = res and res[1]
        spec = dir(spec)
    __dict__ = self.__dict__
    __dict__['_spec_class'] = _spec_class
    __dict__['_spec_set'] = spec_set
    __dict__['_spec_signature'] = _spec_signature
    __dict__['_mock_methods'] = spec