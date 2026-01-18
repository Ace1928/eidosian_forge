import os
from filecmp import cmp
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
from pyomo.common.fileutils import this_file_dir
import pyomo.core.expr as EXPR
from pyomo.core.base import SymbolMap
from pyomo.environ import (
from pyomo.repn.plugins.baron_writer import expression_to_string
class XTestBaron_writer(object):
    """These tests verified that the BARON writer complained loudly for
    variables that were not on the model, not on an active block, or not
    on a Block ctype.  As we are relaxing that requirement throughout
    Pyomo, these tests have been disabled."""

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace('test_', '', 1))
        return (prefix + '.bar.baseline', prefix + '.bar.out')

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(KeyError, model.write, test_fname, format='bar')
        self._cleanup(test_fname)

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(KeyError, model.write, test_fname, format='bar')
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
        self.assertRaises(KeyError, model.write, test_fname, format='bar')
        self._cleanup(test_fname)