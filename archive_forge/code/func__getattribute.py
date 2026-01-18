import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
def _getattribute(self, name):
    """Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattribute__(). It will be added into
    the extended python module as a callback to reduce API overhead.
    """
    func__fastdict_insert = object.__getattribute__(self, '_fastdict_insert')
    if name == 'lite':
        if self._tfmw_has_lite:
            attr = self._tfmw_import_module(name)
            setattr(self._tfmw_wrapped_module, 'lite', attr)
            func__fastdict_insert(name, attr)
            return attr
    attr = object.__getattribute__(self, name)
    if name.startswith('__') or name.startswith('_tfmw_') or name.startswith('_fastdict_'):
        func__fastdict_insert(name, attr)
        return attr
    if not (self._tfmw_print_deprecation_warnings and self._tfmw_add_deprecation_warning(name, attr)):
        func__fastdict_insert(name, attr)
    return attr