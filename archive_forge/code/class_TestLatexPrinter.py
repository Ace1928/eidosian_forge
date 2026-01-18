import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
class TestLatexPrinter(unittest.TestCase):

    def test_latexPrinter_simpleDocTests(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        pstr = latex_printer(m.x + m.y)
        bstr = dedent('\n        \\begin{equation} \n             x + y \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.expression_1 = pyo.Expression(expr=m.x ** 2 + m.y ** 2)
        pstr = latex_printer(m.expression_1)
        bstr = dedent('\n        \\begin{equation} \n             x^{2} + y^{2} \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.constraint_1 = pyo.Constraint(expr=m.x ** 2 + m.y ** 2 <= 1.0)
        pstr = latex_printer(m.constraint_1)
        bstr = dedent('\n        \\begin{equation} \n             x^{2} + y^{2} \\leq 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum((m.v[i] for i in m.I)) <= 0
        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in I  } v_{i} \\leq 0 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Param(initialize=1.0, mutable=True)
        m.objective = pyo.Objective(expr=m.x + m.y + m.z)
        m.constraint_1 = pyo.Constraint(expr=m.x ** 2 + m.y ** 2.0 - m.z ** 2.0 <= m.c)
        pstr = latex_printer(m)
        bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x + y + z & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{s.t.} \n            & & x^{2} + y^{2} - z^{2} \\leq c & \\label{con:basicFormulation_constraint_1} \n        \\end{align} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum((m.v[i] for i in m.I)) <= 0
        m.constraint = pyo.Constraint(rule=ruleMaker)
        lcm = ComponentMap()
        lcm[m.v] = 'x'
        lcm[m.I] = ['\\mathcal{A}', ['j', 'k']]
        pstr = latex_printer(m.constraint, latex_component_map=lcm)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ j \\in \\mathcal{A}  } x_{j} \\leq 0 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_objective(self):
        m = generate_model()
        pstr = latex_printer(m.objective_1)
        bstr = dedent('\n        \\begin{equation} \n            & \\min \n            & & x + y + z \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        pstr = latex_printer(m.objective_3)
        bstr = dedent('\n        \\begin{equation} \n            & \\max \n            & & x + y + z \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_constraint(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_1)
        bstr = dedent('\n        \\begin{equation} \n             x^{2} + y^{-2} - x y z + 1 = 2 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_expression(self):
        m = generate_model()
        m.express = pyo.Expression(expr=m.x + m.y)
        pstr = latex_printer(m.express)
        bstr = dedent('\n        \\begin{equation} \n             x + y \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_simpleExpression(self):
        m = generate_model()
        pstr = latex_printer(m.x - m.y)
        bstr = dedent('\n        \\begin{equation} \n             x - y \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        pstr = latex_printer(m.x - 2 * m.y)
        bstr = dedent('\n        \\begin{equation} \n             x - 2 y \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_unary(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_2)
        bstr = dedent('\n        \\begin{equation} \n              \\left| \\frac{x}{z^{-2}} \\right|   \\left( x + y \\right)  \\leq 2 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        pstr = latex_printer(pyo.Constraint(expr=pyo.sin(m.x) == 1))
        bstr = dedent('\n        \\begin{equation} \n             \\sin \\left( x \\right)  = 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        pstr = latex_printer(pyo.Constraint(expr=pyo.log10(m.x) == 1))
        bstr = dedent('\n        \\begin{equation} \n             \\log_{10} \\left( x \\right)  = 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        pstr = latex_printer(pyo.Constraint(expr=pyo.sqrt(m.x) == 1))
        bstr = dedent('\n        \\begin{equation} \n             \\sqrt { x } = 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_rangedConstraint(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_4)
        bstr = dedent('\n        \\begin{equation} \n             1 \\leq x \\leq 2 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_exprIf(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_5)
        bstr = dedent('\n        \\begin{equation} \n             f_{\\text{exprIf}}(x \\leq 1,z,y) \\leq 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_blackBox(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_6)
        bstr = dedent('\n        \\begin{equation} \n             x + f\\_1(x,y) = 2 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_iteratedConstraints(self):
        m = generate_model()
        pstr = latex_printer(m.constraint_7)
        bstr = dedent('\n        \\begin{equation} \n              \\left( x + y \\right)  \\sum_{ i \\in I  } v_{i} + u_{i,j}^{2} \\leq 0  \\qquad \\forall j \\in I \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        pstr = latex_printer(m.constraint_8)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in K  } p_{i} = 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_fileWriter(self):
        m = generate_simple_model()
        with TempfileManager.new_context() as tempfile:
            fd, fname = tempfile.mkstemp()
            pstr = latex_printer(m, ostream=fname)
            f = open(fname)
            bstr = f.read()
            f.close()
            bstr_split = bstr.split('\n')
            bstr_stripped = bstr_split[8:-2]
            bstr = '\n'.join(bstr_stripped) + '\n'
            self.assertEqual(pstr + '\n', bstr)

    def test_latexPrinter_inputError(self):
        self.assertRaises(ValueError, latex_printer, **{'pyomo_component': 'errorString'})

    def test_latexPrinter_fileWriter(self):
        m = generate_simple_model()
        with TempfileManager.new_context() as tempfile:
            fd, fname = tempfile.mkstemp()
            pstr = latex_printer(m, ostream=fname)
            f = open(fname)
            bstr = f.read()
            f.close()
            bstr_split = bstr.split('\n')
            bstr_stripped = bstr_split[8:-2]
            bstr = '\n'.join(bstr_stripped) + '\n'
            self.assertEqual(pstr + '\n', bstr)
        self.assertRaises(ValueError, latex_printer, **{'pyomo_component': m, 'ostream': 2.0})

    def test_latexPrinter_overwriteError(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum((m.v[i] for i in m.I)) <= 0
        m.constraint = pyo.Constraint(rule=ruleMaker)
        lcm = ComponentMap()
        lcm[m.v] = 'x'
        lcm[m.I] = ['\\mathcal{A}', ['j', 'k']]
        lcm['err'] = 1.0
        self.assertRaises(ValueError, latex_printer, **{'pyomo_component': m.constraint, 'latex_component_map': lcm})

    def test_latexPrinter_indexedParam(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.x = pyo.Var(m.I * m.I)
        m.c = pyo.Param(m.I * m.I, initialize=1.0, mutable=True)

        def ruleMaker_1(m):
            return sum((m.c[i, j] * m.x[i, j] for i in m.I for j in m.I))

        def ruleMaker_2(m):
            return sum((m.x[i, j] ** 2 for i in m.I for j in m.I)) <= 1
        m.objective = pyo.Objective(rule=ruleMaker_1)
        m.constraint_1 = pyo.Constraint(rule=ruleMaker_2)
        pstr = latex_printer(m)
        bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & \\sum_{ i \\in I  } \\sum_{ j \\in I  } c_{i,j} x_{i,j} & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{s.t.} \n            & & \\sum_{ i \\in I  } \\sum_{ j \\in I  } x_{i,j}^{2} \\leq 1 & \\label{con:basicFormulation_constraint_1} \n        \\end{align} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)
        lcm = ComponentMap()
        lcm[m.I] = ['\\mathcal{A}', ['j']]
        self.assertRaises(ValueError, latex_printer, **{'pyomo_component': m, 'latex_component_map': lcm})

    def test_latexPrinter_involvedModel(self):
        m = generate_model()
        pstr = latex_printer(m)
        print(pstr)
        bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x + y + z & \\label{obj:basicFormulation_objective_1} \\\\ \n            & \\min \n            & &  \\left( x + y \\right)  \\sum_{ i \\in J  } w_{i} & \\label{obj:basicFormulation_objective_2} \\\\ \n            & \\max \n            & & x + y + z & \\label{obj:basicFormulation_objective_3} \\\\ \n            & \\text{s.t.} \n            & & x^{2} + y^{-2} - x y z + 1 = 2 & \\label{con:basicFormulation_constraint_1} \\\\ \n            &&&  \\left| \\frac{x}{z^{-2}} \\right|   \\left( x + y \\right)  \\leq 2 & \\label{con:basicFormulation_constraint_2} \\\\ \n            &&& \\sqrt { \\frac{x}{z^{-2}} } \\leq 2 & \\label{con:basicFormulation_constraint_3} \\\\ \n            &&& 1 \\leq x \\leq 2 & \\label{con:basicFormulation_constraint_4} \\\\ \n            &&& f_{\\text{exprIf}}(x \\leq 1,z,y) \\leq 1 & \\label{con:basicFormulation_constraint_5} \\\\ \n            &&& x + f\\_1(x,y) = 2 & \\label{con:basicFormulation_constraint_6} \\\\ \n            &&&  \\left( x + y \\right)  \\sum_{ i \\in I  } v_{i} + u_{i,j}^{2} \\leq 0 &  \\qquad \\forall j \\in I \\label{con:basicFormulation_constraint_7} \\\\ \n            &&& \\sum_{ i \\in K  } p_{i} = 1 & \\label{con:basicFormulation_constraint_8} \n        \\end{align} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_continuousSet(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum((m.v[i] for i in m.I)) <= 0
        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint, explicit_set_summation=True)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i = 1 }^{5} v_{i} \\leq 0 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_notContinuousSet(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum((m.v[i] for i in m.I)) <= 0
        m.constraint = pyo.Constraint(rule=ruleMaker)
        pstr = latex_printer(m.constraint, explicit_set_summation=True)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in I  } v_{i} \\leq 0 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_autoIndex(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.v = pyo.Var(m.I)

        def ruleMaker(m):
            return sum((m.v[i] for i in m.I)) <= 0
        m.constraint = pyo.Constraint(rule=ruleMaker)
        lcm = ComponentMap()
        lcm[m.v] = 'x'
        lcm[m.I] = ['\\mathcal{A}', []]
        pstr = latex_printer(m.constraint, latex_component_map=lcm)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in \\mathcal{A}  } x_{i} \\leq 0 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_equationEnvironment(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Param(initialize=1.0, mutable=True)
        m.objective = pyo.Objective(expr=m.x + m.y + m.z)
        m.constraint_1 = pyo.Constraint(expr=m.x ** 2 + m.y ** 2.0 - m.z ** 2.0 <= m.c)
        pstr = latex_printer(m, use_equation_environment=True)
        bstr = dedent('\n        \\begin{equation} \n            \\begin{aligned} \n                & \\min \n                & & x + y + z \\\\ \n                & \\text{s.t.} \n                & & x^{2} + y^{2} - z^{2} \\leq c \n            \\end{aligned} \n            \\label{basicFormulation} \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_manyVariablesWithDomains(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, 10))
        m.y = pyo.Var(domain=Binary, bounds=(-10, 10))
        m.z = pyo.Var(domain=PositiveReals, bounds=(-10, 10))
        m.u = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 10))
        m.v = pyo.Var(domain=NegativeReals, bounds=(-10, 10))
        m.w = pyo.Var(domain=PercentFraction, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x + m.y + m.z + m.u + m.v + m.w)
        pstr = latex_printer(m)
        bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x + y + z + u + v + w & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{w.b.} \n            & & -10 \\leq x \\leq 10 & \\qquad \\in \\mathds{Z} \\label{con:basicFormulation_x_bound} \\\\ \n            &&& y & \\qquad \\in \\left\\{ 0 , 1 \\right \\} \\label{con:basicFormulation_y_bound} \\\\ \n            &&&  0 < z \\leq 10 & \\qquad \\in \\mathds{R}_{> 0} \\label{con:basicFormulation_z_bound} \\\\ \n            &&&  0 \\leq u \\leq 10 & \\qquad \\in \\mathds{Z}_{\\geq 0} \\label{con:basicFormulation_u_bound} \\\\ \n            &&& -10 \\leq v < 0  & \\qquad \\in \\mathds{R}_{< 0} \\label{con:basicFormulation_v_bound} \\\\ \n            &&&  0 \\leq w \\leq 1  & \\qquad \\in \\mathds{R} \\label{con:basicFormulation_w_bound} \n        \\end{align} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_manyVariablesWithDomains_eqn(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.x = pyo.Var(domain=Integers, bounds=(-10, 10))
        m.y = pyo.Var(domain=Binary, bounds=(-10, 10))
        m.z = pyo.Var(domain=PositiveReals, bounds=(-10, 10))
        m.u = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 10))
        m.v = pyo.Var(domain=NegativeReals, bounds=(-10, 10))
        m.w = pyo.Var(domain=PercentFraction, bounds=(-10, 10))
        m.objective = pyo.Objective(expr=m.x + m.y + m.z + m.u + m.v + m.w)
        pstr = latex_printer(m, use_equation_environment=True)
        bstr = dedent('\n        \\begin{equation} \n            \\begin{aligned} \n                & \\min \n                & & x + y + z + u + v + w \\\\ \n                & \\text{w.b.} \n                & & -10 \\leq x \\leq 10 \\qquad \\in \\mathds{Z}\\\\ \n                &&& y \\qquad \\in \\left\\{ 0 , 1 \\right \\}\\\\ \n                &&&  0 < z \\leq 10 \\qquad \\in \\mathds{R}_{> 0}\\\\ \n                &&&  0 \\leq u \\leq 10 \\qquad \\in \\mathds{Z}_{\\geq 0}\\\\ \n                &&& -10 \\leq v < 0  \\qquad \\in \\mathds{R}_{< 0}\\\\ \n                &&&  0 \\leq w \\leq 1  \\qquad \\in \\mathds{R}\n            \\end{aligned} \n            \\label{basicFormulation} \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_indexedParamSingle(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.x = pyo.Var(m.I * m.I)
        m.c = pyo.Param(m.I * m.I, initialize=1.0, mutable=True)

        def ruleMaker_1(m):
            return sum((m.c[i, j] * m.x[i, j] for i in m.I for j in m.I))

        def ruleMaker_2(m):
            return sum((m.c[i, j] * m.x[i, j] ** 2 for i in m.I for j in m.I)) <= 1
        m.objective = pyo.Objective(rule=ruleMaker_1)
        m.constraint_1 = pyo.Constraint(rule=ruleMaker_2)
        pstr = latex_printer(m.constraint_1)
        print(pstr)
        bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in I  } \\sum_{ j \\in I  } c_{i,j} x_{i,j}^{2} \\leq 1 \n        \\end{equation} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)

    def test_latexPrinter_throwTemplatizeError(self):
        m = pyo.ConcreteModel(name='basicFormulation')
        m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
        m.x = pyo.Var(m.I, bounds=[-10, 10])
        m.c = pyo.Param(m.I, initialize=1.0, mutable=True)

        def ruleMaker_1(m):
            return sum((m.c[i] * m.x[i] for i in m.I))

        def ruleMaker_2(m, i):
            if i >= 2:
                return m.x[i] <= 1
            else:
                return pyo.Constraint.Skip
        m.objective = pyo.Objective(rule=ruleMaker_1)
        m.constraint_1 = pyo.Constraint(m.I, rule=ruleMaker_2)
        self.assertRaises(RuntimeError, latex_printer, **{'pyomo_component': m, 'throw_templatization_error': True})
        pstr = latex_printer(m)
        bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & \\sum_{ i \\in I  } c_{i} x_{i} & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{s.t.} \n            & & x[2] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & & x[3] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & & x[4] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & & x[5] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & \\text{w.b.} \n            & & -10 \\leq x \\leq 10 & \\qquad \\in \\mathds{R} \\label{con:basicFormulation_x_bound} \n        \\end{align} \n        ')
        self.assertEqual('\n' + pstr + '\n', bstr)