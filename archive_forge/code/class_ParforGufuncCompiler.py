import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
class ParforGufuncCompiler(compiler.CompilerBase):

    def define_pipelines(self):
        from numba.core.compiler_machinery import PassManager
        dpb = compiler.DefaultPassBuilder
        pm = PassManager('full_parfor_gufunc')
        parfor_gufunc_passes = dpb.define_parfor_gufunc_pipeline(self.state)
        pm.passes.extend(parfor_gufunc_passes.passes)
        lowering_passes = dpb.define_parfor_gufunc_nopython_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)
        pm.finalize()
        return [pm]