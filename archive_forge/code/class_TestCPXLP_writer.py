import os
import random
from ..lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap
class TestCPXLP_writer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = TempfileManager.new_context()
        cls.tempdir = cls.context.create_tempdir()

    @classmethod
    def tearDownClass(cls):
        cls.context.release(remove=False)

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = test_name.replace('test_', '', 1)
        return (os.path.join(thisdir, prefix + '.lp.baseline'), os.path.join(self.tempdir, prefix + '.lp.out'))

    def test_linear_var_on_other_model(self):
        baseline_fname, test_fname = self._get_fnames()
        other = ConcreteModel()
        other.a = Var()
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.c = Constraint(expr=other.a + 2 * model.x <= 0)
        with LoggingIntercept() as LOG:
            self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')
        self.assertEqual(LOG.getvalue().replace('\n', ' ').strip(), 'Model contains an expression (c) that contains a variable (a) that is not attached to an active block on the submodel being written')
        model.write(test_fname, format='lp_v2')
        self.assertEqual(*load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2'))

    def test_quadratic_var_on_other_model(self):
        baseline_fname, test_fname = self._get_fnames()
        other = ConcreteModel()
        other.a = Var()
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.c = Constraint(expr=other.a * model.x <= 0)
        with LoggingIntercept() as LOG:
            self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')
        self.assertEqual(LOG.getvalue().replace('\n', ' ').strip(), 'Model contains an expression (c) that contains a variable (a) that is not attached to an active block on the submodel being written')
        model.write(test_fname, format='lp_v2')
        self.assertEqual(*load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2'))

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)
        baseline_fname, test_fname = self._get_fnames()
        model.write(test_fname, format='lp_v1')
        self.assertEqual(*load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v1'))
        model.write(test_fname, format='lp_v2')
        self.assertEqual(*load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2'))

    def test_var_on_nonblock(self):

        class Foo(Block().__class__):

            def __init__(self, *args, **kwds):
                kwds.setdefault('ctype', Foo)
                super(Foo, self).__init__(*args, **kwds)
        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.deactivate()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)
        baseline_fname, test_fname = self._get_fnames()
        self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')
        model.write(test_fname, format='lp_v2')
        self.assertEqual(*load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2'))

    def test_obj_con_cache(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x >= 1)
        model.obj = Objective(expr=model.x * 2)
        with TempfileManager.new_context() as TMP:
            lp_file = TMP.create_tempfile(suffix='.lp')
            model.write(lp_file, format='lp_v1')
            self.assertFalse(hasattr(model, '_repn'))
            with open(lp_file) as FILE:
                lp_ref = FILE.read()
            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = True
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 1)
            self.assertIn(model.obj, model._repn)
            obj_repn = model._repn[model.obj]
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)
            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = None
            model._gen_con_repn = True
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            obj_repn = model._repn[model.obj]
            c_repn = model._repn[model.c]
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)
            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = None
            model._gen_con_repn = None
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)
            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = True
            model._gen_con_repn = True
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIsNot(obj_repn, model._repn[model.obj])
            self.assertIsNot(c_repn, model._repn[model.c])
            obj_repn = model._repn[model.obj]
            c_repn = model._repn[model.c]
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)
            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = False
            model._gen_con_repn = False
            import pyomo.repn.plugins.ampl.ampl_ as ampl_
            gsr = ampl_.generate_standard_repn
            try:

                def dont_call_gsr(*args, **kwargs):
                    self.fail('generate_standard_repn should not be called')
                ampl_.generate_standard_repn = dont_call_gsr
                model.write(lp_file, format='lp_v1')
            finally:
                ampl_.generate_standard_repn = gsr
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)