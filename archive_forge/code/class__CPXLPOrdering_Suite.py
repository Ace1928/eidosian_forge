import os
import random
from ..lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap
class _CPXLPOrdering_Suite(object):

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

    def _check_baseline(self, model, **kwds):
        baseline, testfile = self._get_fnames()
        io_options = {'symbolic_solver_labels': True}
        io_options.update(kwds)
        model.write(testfile, format=self._lp_version, io_options=io_options)
        self.assertEqual(*load_and_compare_lp_baseline(baseline, testfile, self._lp_version))

    def _gen_expression(self, terms):
        expr = 0.0
        for term in terms:
            if type(term) is tuple:
                prodexpr = 1.0
                for x in term:
                    prodexpr *= x
                expr += prodexpr
            else:
                expr += term
        return expr

    def test_no_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()
        terms = [model.a, model.b, model.c, (model.a, model.a), (model.b, model.b), (model.c, model.c), (model.a, model.b), (model.a, model.c), (model.b, model.c)]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()
        terms = [model.a, model.b, model.c, (model.a, model.a), (model.b, model.b), (model.c, model.c), (model.a, model.b), (model.a, model.c), (model.b, model.c)]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        column_order = ComponentMap()
        column_order[model.a] = 2
        column_order[model.b] = 1
        column_order[model.c] = 0
        self._check_baseline(model, column_order=column_order)

    def test_no_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()
        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()
        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        column_order = ComponentMap()
        column_order[model.a] = 2
        column_order[model.b] = 1
        column_order[model.c] = 0
        self._check_baseline(model, column_order=column_order)

    def test_no_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()
        components = {}
        components['obj'] = Objective(expr=model.a)
        components['con1'] = Constraint(expr=model.a >= 0)
        components['con2'] = Constraint(expr=model.a <= 1)
        components['con3'] = Constraint(expr=(0, model.a, 1))
        components['con4'] = Constraint([1, 2], rule=lambda m, i: model.a == i)
        random_order = list(components.keys())
        random.shuffle(random_order)
        for key in random_order:
            model.add_component(key, components[key])
        self._check_baseline(model, file_determinism=2)

    def test_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()
        components = {}
        components['obj'] = Objective(expr=model.a)
        components['con1'] = Constraint(expr=model.a >= 0)
        components['con2'] = Constraint(expr=model.a <= 1)
        components['con3'] = Constraint(expr=(0, model.a, 1))
        components['con4'] = Constraint([1, 2], rule=lambda m, i: model.a == i)
        random_order = list(components.keys())
        random.shuffle(random_order)
        for key in random_order:
            model.add_component(key, components[key])
        row_order = ComponentMap()
        row_order[model.con1] = 100
        row_order[model.con2] = 2
        row_order[model.con3] = 1
        row_order[model.con4[1]] = 0
        row_order[model.con4[2]] = -1
        self._check_baseline(model, row_order=row_order)