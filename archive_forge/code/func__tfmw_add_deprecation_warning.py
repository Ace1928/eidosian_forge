import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
def _tfmw_add_deprecation_warning(self, name, attr):
    """Print deprecation warning for attr with given name if necessary."""
    if self._tfmw_warning_count < _PER_MODULE_WARNING_LIMIT and name not in self._tfmw_deprecated_checked:
        self._tfmw_deprecated_checked.add(name)
        if self._tfmw_module_name:
            full_name = 'tf.%s.%s' % (self._tfmw_module_name, name)
        else:
            full_name = 'tf.%s' % name
        rename = get_rename_v2(full_name)
        if rename and (not has_deprecation_decorator(attr)):
            call_location = _call_location()
            if not call_location.startswith('<'):
                logging.warning('From %s: The name %s is deprecated. Please use %s instead.\n', _call_location(), full_name, rename)
                self._tfmw_warning_count += 1
                return True
    return False