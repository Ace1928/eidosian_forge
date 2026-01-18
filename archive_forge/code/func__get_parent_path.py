import _imp
import _io
import sys
import _warnings
import marshal
def _get_parent_path(self):
    parent_module_name, path_attr_name = self._find_parent_path_names()
    return getattr(sys.modules[parent_module_name], path_attr_name)