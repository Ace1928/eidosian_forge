import os
import re
import six
from six.moves import urllib
from routes import request_config
def _screenargs(kargs, mapper, environ, force_explicit=False):
    """
    Private function that takes a dict, and screens it against the current
    request dict to determine what the dict should look like that is used.
    This is responsible for the requests "memory" of the current.
    """
    encoding = mapper.encoding
    for key, val in six.iteritems(kargs):
        if isinstance(val, six.text_type):
            kargs[key] = val.encode(encoding)
    if mapper.explicit and mapper.sub_domains and (not force_explicit):
        return _subdomain_check(kargs, mapper, environ)
    elif mapper.explicit and (not force_explicit):
        return kargs
    controller_name = as_unicode(kargs.get('controller'), encoding)
    if controller_name and controller_name.startswith('/'):
        kargs['controller'] = kargs['controller'][1:]
        return kargs
    elif controller_name and 'action' not in kargs:
        kargs['action'] = 'index'
    route_args = environ.get('wsgiorg.routing_args')
    if route_args:
        memory_kargs = route_args[1].copy()
    else:
        memory_kargs = {}
    empty_keys = [key for key, value in six.iteritems(kargs) if value is None]
    for key in empty_keys:
        del kargs[key]
        memory_kargs.pop(key, None)
    memory_kargs.update(kargs)
    if mapper.sub_domains:
        memory_kargs = _subdomain_check(memory_kargs, mapper, environ)
    return memory_kargs