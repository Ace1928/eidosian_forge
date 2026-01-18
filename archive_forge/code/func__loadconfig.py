from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _loadconfig(object_type, uri, path, name, relative_to, global_conf):
    isabs = os.path.isabs(path)
    path = path.replace('\\', '/')
    if not isabs:
        if not relative_to:
            raise ValueError('Cannot resolve relative uri %r; no relative_to keyword argument given' % uri)
        relative_to = relative_to.replace('\\', '/')
        if relative_to.endswith('/'):
            path = relative_to + path
        else:
            path = relative_to + '/' + path
    if path.startswith('///'):
        path = path[2:]
    path = unquote(path)
    loader = ConfigLoader(path)
    if global_conf:
        loader.update_defaults(global_conf, overwrite=False)
    return loader.get_context(object_type, name, global_conf)