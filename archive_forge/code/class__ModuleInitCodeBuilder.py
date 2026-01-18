import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
class _ModuleInitCodeBuilder(object):
    """Builds a map from module name to imports included in that module."""

    def __init__(self, output_package, api_version, lazy_loading=_LAZY_LOADING, use_relative_imports=False):
        self._output_package = output_package
        self._module_imports = collections.defaultdict(lambda: collections.defaultdict(set))
        self._dest_import_to_id = collections.defaultdict(int)
        self._underscore_names_in_root = set()
        self._api_version = api_version
        self._lazy_loading = lazy_loading
        self._use_relative_imports = use_relative_imports

    def _check_already_imported(self, symbol_id, api_name):
        if api_name in self._dest_import_to_id and symbol_id != self._dest_import_to_id[api_name] and (symbol_id != -1):
            raise SymbolExposedTwiceError(f'Trying to export multiple symbols with same name: {api_name}')
        self._dest_import_to_id[api_name] = symbol_id

    def add_import(self, symbol, source_module_name, source_name, dest_module_name, dest_name):
        """Adds this import to module_imports.

    Args:
      symbol: TensorFlow Python symbol.
      source_module_name: (string) Module to import from.
      source_name: (string) Name of the symbol to import.
      dest_module_name: (string) Module name to add import to.
      dest_name: (string) Import the symbol using this name.

    Raises:
      SymbolExposedTwiceError: Raised when an import with the same
        dest_name has already been added to dest_module_name.
    """
        if source_module_name.endswith('python.modules_with_exports'):
            source_module_name = symbol.__module__
        import_str = self.format_import(source_module_name, source_name, dest_name)
        full_api_name = dest_name
        if dest_module_name:
            full_api_name = dest_module_name + '.' + full_api_name
        symbol_id = -1 if not symbol else id(symbol)
        self._check_already_imported(symbol_id, full_api_name)
        if not dest_module_name and dest_name.startswith('_'):
            self._underscore_names_in_root.add(dest_name)
        priority = 0
        if symbol:
            if hasattr(symbol, '__module__'):
                priority = int(source_module_name == symbol.__module__)
            if hasattr(symbol, '__name__'):
                priority += int(source_name == symbol.__name__)
        self._module_imports[dest_module_name][full_api_name].add((import_str, priority))

    def _import_submodules(self):
        """Add imports for all destination modules in self._module_imports."""
        imported_modules = set(self._module_imports.keys())
        for module in imported_modules:
            if not module:
                continue
            module_split = module.split('.')
            parent_module = ''
            for submodule_index in range(len(module_split)):
                if submodule_index > 0:
                    submodule = module_split[submodule_index - 1]
                    parent_module += '.' + submodule if parent_module else submodule
                import_from = self._output_package
                if self._lazy_loading:
                    import_from += '.' + '.'.join(module_split[:submodule_index + 1])
                    self.add_import(symbol=None, source_module_name='', source_name=import_from, dest_module_name=parent_module, dest_name=module_split[submodule_index])
                else:
                    if self._use_relative_imports:
                        import_from = '.'
                    elif submodule_index > 0:
                        import_from += '.' + '.'.join(module_split[:submodule_index])
                    self.add_import(symbol=None, source_module_name=import_from, source_name=module_split[submodule_index], dest_module_name=parent_module, dest_name=module_split[submodule_index])

    def build(self):
        """Get a map from destination module to __init__.py code for that module.

    Returns:
      A dictionary where
        key: (string) destination module (for e.g. tf or tf.consts).
        value: (string) text that should be in __init__.py files for
          corresponding modules.
    """
        self._import_submodules()
        module_text_map = {}
        footer_text_map = {}
        for dest_module, dest_name_to_imports in self._module_imports.items():
            imports_list = [get_canonical_import(imports) for _, imports in dest_name_to_imports.items()]
            if self._lazy_loading:
                module_text_map[dest_module] = _LAZY_LOADING_MODULE_TEXT_TEMPLATE % '\n'.join(sorted(imports_list))
            else:
                module_text_map[dest_module] = '\n'.join(sorted(imports_list))
        root_module_footer = ''
        if not self._lazy_loading:
            underscore_names_str = ', '.join(("'%s'" % name for name in sorted(self._underscore_names_in_root)))
            root_module_footer = "\n_names_with_underscore = [%s]\n__all__ = [_s for _s in dir() if not _s.startswith('_')]\n__all__.extend([_s for _s in _names_with_underscore])\n" % underscore_names_str
        if self._api_version == 1 or self._lazy_loading:
            for dest_module, _ in self._module_imports.items():
                deprecation = 'False'
                has_lite = 'False'
                if self._api_version == 1:
                    if not dest_module.startswith(_COMPAT_MODULE_PREFIX):
                        deprecation = 'True'
                if not dest_module and 'lite' in self._module_imports and self._lazy_loading:
                    has_lite = 'True'
                if self._lazy_loading:
                    public_apis_name = '_PUBLIC_APIS'
                else:
                    public_apis_name = 'None'
                footer_text_map[dest_module] = _DEPRECATION_FOOTER % (dest_module, public_apis_name, deprecation, has_lite)
        return (module_text_map, footer_text_map, root_module_footer)

    def format_import(self, source_module_name, source_name, dest_name):
        """Formats import statement.

    Args:
      source_module_name: (string) Source module to import from.
      source_name: (string) Source symbol name to import.
      dest_name: (string) Destination alias name.

    Returns:
      An import statement string.
    """
        if self._lazy_loading:
            return "  '%s': ('%s', '%s')," % (dest_name, source_module_name, source_name)
        elif source_module_name:
            if source_name == dest_name:
                return 'from %s import %s' % (source_module_name, source_name)
            else:
                return 'from %s import %s as %s' % (source_module_name, source_name, dest_name)
        elif source_name == dest_name:
            return 'import %s' % source_name
        else:
            return 'import %s as %s' % (source_name, dest_name)

    def get_destination_modules(self):
        return set(self._module_imports.keys())

    def copy_imports(self, from_dest_module, to_dest_module):
        self._module_imports[to_dest_module] = self._module_imports[from_dest_module].copy()