import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
class TestGams_writer(unittest.TestCase):

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace('test_', '', 1))
        return (prefix + '.gams.baseline', prefix + '.gams.out')

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(RuntimeError, model.write, test_fname, format='gams')
        self._cleanup(test_fname)

    def test_var_on_nonblock(self):

        class Foo(Block().__class__):

            def __init__(self, *args, **kwds):
                kwds.setdefault('ctype', Foo)
                super(Foo, self).__init__(*args, **kwds)
        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(RuntimeError, model.write, test_fname, format='gams')
        self._cleanup(test_fname)