import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
def _deqv(m, i):
    return m.v[i] ** 2 + m.v[i] == -m.dv[i]