import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
class TFModuleWrapper(FastModuleType):
    """Wrapper for TF modules to support deprecation messages and lazyloading."""
    compat_v1_usage_recorded = False

    def __init__(self, wrapped, module_name, public_apis=None, deprecation=True, has_lite=False):
        super(TFModuleWrapper, self).__init__(wrapped.__name__)
        FastModuleType.set_getattr_callback(self, TFModuleWrapper._getattr)
        FastModuleType.set_getattribute_callback(self, TFModuleWrapper._getattribute)
        self.__dict__.update(wrapped.__dict__)
        self._tfmw_wrapped_module = wrapped
        self._tfmw_module_name = module_name
        self._tfmw_public_apis = public_apis
        self._tfmw_print_deprecation_warnings = deprecation
        self._tfmw_has_lite = has_lite
        self._tfmw_is_compat_v1 = wrapped.__name__.endswith('.compat.v1')
        if self._tfmw_public_apis:
            self._tfmw_wrapped_module.__all__ = list(self._tfmw_public_apis.keys())
            self.__all__ = list(self._tfmw_public_apis.keys())
        elif hasattr(self._tfmw_wrapped_module, '__all__'):
            self.__all__ = self._tfmw_wrapped_module.__all__
        else:
            self._tfmw_wrapped_module.__all__ = [attr for attr in dir(self._tfmw_wrapped_module) if not attr.startswith('_')]
            self.__all__ = self._tfmw_wrapped_module.__all__
        self._tfmw_deprecated_checked = set()
        self._tfmw_warning_count = 0

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

    def _tfmw_import_module(self, name):
        """Lazily loading the modules."""
        if self._tfmw_is_compat_v1 and name != 'app' and (not TFModuleWrapper.compat_v1_usage_recorded):
            TFModuleWrapper.compat_v1_usage_recorded = True
            compat_v1_usage_gauge.get_cell().set(True)
        symbol_loc_info = self._tfmw_public_apis[name]
        if symbol_loc_info[0]:
            module = importlib.import_module(symbol_loc_info[0])
            attr = getattr(module, symbol_loc_info[1])
        else:
            attr = importlib.import_module(symbol_loc_info[1])
        setattr(self._tfmw_wrapped_module, name, attr)
        self.__dict__[name] = attr
        self._fastdict_insert(name, attr)
        return attr

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

    def _getattr(self, name):
        """Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattr__(). It will be added into the
    extended python module as a callback to reduce API overhead. Instead of
    relying on implicit AttributeError handling, this added callback function
    will
    be called explicitly from the extended C API if the default attribute lookup
    fails.
    """
        try:
            attr = getattr(self._tfmw_wrapped_module, name)
        except AttributeError:
            if not self._tfmw_public_apis:
                raise
            if name not in self._tfmw_public_apis:
                raise
            attr = self._tfmw_import_module(name)
        if self._tfmw_print_deprecation_warnings:
            self._tfmw_add_deprecation_warning(name, attr)
        return attr

    def __setattr__(self, arg, val):
        if not arg.startswith('_tfmw_'):
            setattr(self._tfmw_wrapped_module, arg, val)
            self.__dict__[arg] = val
            if arg not in self.__all__ and arg != '__all__':
                self.__all__.append(arg)
            if self._fastdict_key_in(arg):
                self._fastdict_insert(arg, val)
        super(TFModuleWrapper, self).__setattr__(arg, val)

    def __dir__(self):
        if self._tfmw_public_apis:
            return list(set(self._tfmw_public_apis.keys()).union(set([attr for attr in dir(self._tfmw_wrapped_module) if not attr.startswith('_')])))
        else:
            return dir(self._tfmw_wrapped_module)

    def __delattr__(self, name):
        if name.startswith('_tfmw_'):
            super(TFModuleWrapper, self).__delattr__(name)
        else:
            delattr(self._tfmw_wrapped_module, name)

    def __repr__(self):
        return self._tfmw_wrapped_module.__repr__()

    def __reduce__(self):
        return (importlib.import_module, (self.__name__,))