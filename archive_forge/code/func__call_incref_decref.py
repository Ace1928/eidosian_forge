import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def _call_incref_decref(self, builder, typ, value, funcname):
    """Call function of *funcname* on every meminfo found in *value*.
        """
    self._require_nrt()
    from numba.core.runtime.nrtdynmod import incref_decref_ty
    meminfos = self.get_meminfos(builder, typ, value)
    for _, mi in meminfos:
        mod = builder.module
        fn = cgutils.get_or_insert_function(mod, incref_decref_ty, funcname)
        fn.args[0].add_attribute('noalias')
        fn.args[0].add_attribute('nocapture')
        builder.call(fn, [mi])