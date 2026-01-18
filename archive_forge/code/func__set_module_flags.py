import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _set_module_flags(self):
    """Set the module flags metadata
        """
    module = self.module
    mflags = cgutils.get_or_insert_named_metadata(module, 'llvm.module.flags')
    require_warning_behavior = self._const_int(2)
    if self.DWARF_VERSION is not None:
        dwarf_version = module.add_metadata([require_warning_behavior, 'Dwarf Version', self._const_int(self.DWARF_VERSION)])
        if dwarf_version not in mflags.operands:
            mflags.add(dwarf_version)
    debuginfo_version = module.add_metadata([require_warning_behavior, 'Debug Info Version', self._const_int(self.DEBUG_INFO_VERSION)])
    if debuginfo_version not in mflags.operands:
        mflags.add(debuginfo_version)