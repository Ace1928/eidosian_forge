from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class FuncLoader(_Loader):
    """Loader that supports specifying functions inside modules, without
    using eggs at all. Configuration should be in the format:
        use = call:my.module.path:function_name

    Dot notation is supported in both the module and function name, e.g.:
        use = call:my.module.path:object.method
    """

    def __init__(self, spec):
        self.spec = spec
        if ':' not in spec:
            raise LookupError('Configuration not in format module:function')

    def get_context(self, object_type, name=None, global_conf=None):
        obj = lookup_object(self.spec)
        return LoaderContext(obj, object_type, None, global_conf or {}, {}, self)