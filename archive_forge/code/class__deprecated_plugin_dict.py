import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class _deprecated_plugin_dict(dict):

    def __init__(self, name, classdict):
        super().__init__()
        msg = classdict.pop('__deprecated_message__', None)
        if not msg:
            msg = 'The %s interface has been deprecated' % (name,)
        version = classdict.pop('__deprecated_version__', None)
        remove_in = classdict.pop('__deprecated_remove_in__', None)
        self._deprecation_info = {'msg': msg, 'version': version, 'remove_in': remove_in}

    def __setitem__(self, key, val):
        deprecation_warning(**self._deprecation_info)
        super().__setitem__(key, val)

    def items(self):
        deprecation_warning(**self._deprecation_info)
        return super().items()