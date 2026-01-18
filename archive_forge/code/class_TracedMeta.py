import collections
import datetime
import functools
import inspect
import socket
import threading
from oslo_utils import reflection
from oslo_utils import uuidutils
from osprofiler import _utils as utils
from osprofiler import notifier
class TracedMeta(type):
    """Metaclass to comfortably trace all children of a specific class.

    Possible usage:

    >>>  class RpcManagerClass(object, metaclass=profiler.TracedMeta):
    >>>      __trace_args__ = {'name': 'rpc',
    >>>                        'info': None,
    >>>                        'hide_args': False,
    >>>                        'hide_result': True,
    >>>                        'trace_private': False}
    >>>
    >>>      def my_method(self, some_args):
    >>>          pass
    >>>
    >>>      def my_method2(self, some_arg1, some_arg2, kw=None, kw2=None)
    >>>          pass

    Adding of this metaclass requires to set __trace_args__ attribute to the
    class we want to modify. __trace_args__ is the dictionary with one
    mandatory key included - "name", that will define name of action to be
    traced - E.g. wsgi, rpc, db, etc...
    """

    def __init__(cls, cls_name, bases, attrs):
        super(TracedMeta, cls).__init__(cls_name, bases, attrs)
        trace_args = dict(getattr(cls, '__trace_args__', {}))
        trace_private = trace_args.pop('trace_private', False)
        allow_multiple_trace = trace_args.pop('allow_multiple_trace', True)
        if 'name' not in trace_args:
            raise TypeError("Please specify __trace_args__ class level dictionary attribute with mandatory 'name' key - e.g. __trace_args__ = {'name': 'rpc'}")
        traceable_attrs = []
        for attr_name, attr_value in attrs.items():
            if not (inspect.ismethod(attr_value) or inspect.isfunction(attr_value)):
                continue
            if attr_name.startswith('__'):
                continue
            if not trace_private and attr_name.startswith('_'):
                continue
            traceable_attrs.append((attr_name, attr_value))
        if not allow_multiple_trace:
            _ensure_no_multiple_traced(traceable_attrs)
        for attr_name, attr_value in traceable_attrs:
            setattr(cls, attr_name, trace(**trace_args)(getattr(cls, attr_name)))