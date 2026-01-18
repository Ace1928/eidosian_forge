import os
from functools import wraps
import numpy as np
from shapely import lib
from shapely.errors import UnsupportedGEOSVersionError
class requires_geos:

    def __init__(self, version):
        if version.count('.') != 2:
            raise ValueError('Version must be <major>.<minor>.<patch> format')
        self.version = tuple((int(x) for x in version.split('.')))

    def __call__(self, func):
        is_compatible = lib.geos_version >= self.version
        is_doc_build = os.environ.get('SPHINX_DOC_BUILD') == '1'
        if is_compatible and (not is_doc_build):
            return func
        msg = "'{}' requires at least GEOS {}.{}.{}.".format(func.__name__, *self.version)
        if is_compatible:

            @wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)
        else:

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise UnsupportedGEOSVersionError(msg)
        doc = wrapped.__doc__
        if doc:
            position = doc.find('\n\n') + 2
            indent = 0
            while True:
                if doc[position + indent] == ' ':
                    indent += 1
                else:
                    break
            wrapped.__doc__ = doc.replace('\n\n', '\n\n{}.. note:: {}\n\n'.format(' ' * indent, msg), 1)
        return wrapped