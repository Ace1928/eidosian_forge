import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def _set_init(cls):
    """
        Generate a wrapper for calling the constructor from pure Python.
        Note the wrapper will only accept positional arguments.
        """
    init = cls.class_type.instance_type.methods['__init__']
    init_sig = utils.pysignature(init)
    args = _getargs(init_sig)[1:]
    cls._ctor_sig = init_sig
    ctor_source = _ctor_template.format(args=', '.join(args))
    glbls = {'__numba_cls_': cls}
    exec(ctor_source, glbls)
    ctor = glbls['ctor']
    cls._ctor = njit(ctor)