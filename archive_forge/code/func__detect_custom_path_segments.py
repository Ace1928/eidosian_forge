import logging
import re
import warnings
from inspect import getmembers, ismethod
from webob import exc
from .secure import handle_security, cross_boundary
from .util import iscontroller, getargspec, _cfg
def _detect_custom_path_segments(obj):
    if obj.__class__.__module__ == '__builtin__':
        return
    attrs = set(dir(obj))
    if obj.__class__ not in __observed_controllers__:
        for key, val in getmembers(obj):
            if iscontroller(val) and isinstance(getattr(val, 'custom_route', None), str):
                route = val.custom_route
                for conflict in attrs.intersection(set((route,))):
                    raise RuntimeError('%(module)s.%(class)s.%(function)s has a custom path segment, "%(route)s", but %(module)s.%(class)s already has an existing attribute named "%(route)s".' % {'module': obj.__class__.__module__, 'class': obj.__class__.__name__, 'function': val.__name__, 'route': conflict})
                existing = __custom_routes__.get((obj.__class__, route))
                if existing:
                    raise RuntimeError('%(module)s.%(class)s.%(function)s and %(module)s.%(class)s.%(other)s have a conflicting custom path segment, "%(route)s".' % {'module': obj.__class__.__module__, 'class': obj.__class__.__name__, 'function': val.__name__, 'other': existing, 'route': route})
                __custom_routes__[obj.__class__, route] = key
        __observed_controllers__.add(obj.__class__)