import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestUnitsChecking(unittest.TestCase):

    def _create_model_and_vars(self):
        u = units
        m = ConcreteModel()
        m.dx = Var(units=u.m, initialize=0.10188943773836046)
        m.dy = Var(units=u.m, initialize=0.0)
        m.vx = Var(units=u.m / u.s, initialize=0.7071067769802851)
        m.vy = Var(units=u.m / u.s, initialize=0.7071067769802851)
        m.t = Var(units=u.s, bounds=(1e-05, 10.0), initialize=0.0024015570927624456)
        m.theta = Var(bounds=(0, 0.49 * 3.14), initialize=0.7853981693583533, units=u.radians)
        m.a = Param(initialize=-32.2, units=u.ft / u.s ** 2)
        m.x_unitless = Var()
        return m

    def test_assert_units_consistent_equivalent(self):
        u = units
        m = ConcreteModel()
        m.dx = Var(units=u.m, initialize=0.10188943773836046)
        m.dy = Var(units=u.m, initialize=0.0)
        m.vx = Var(units=u.m / u.s, initialize=0.7071067769802851)
        m.vy = Var(units=u.m / u.s, initialize=0.7071067769802851)
        m.t = Var(units=u.min, bounds=(1e-05, 10.0), initialize=0.0024015570927624456)
        m.theta = Var(bounds=(0, 0.49 * 3.14), initialize=0.7853981693583533, units=u.radians)
        m.a = Param(initialize=-32.2, units=u.ft / u.s ** 2)
        m.x_unitless = Var()
        m.obj = Objective(expr=m.dx, sense=maximize)
        m.vx_con = Constraint(expr=m.vx == 1.0 * u.m / u.s * cos(m.theta))
        m.vy_con = Constraint(expr=m.vy == 1.0 * u.m / u.s * sin(m.theta))
        m.dx_con = Constraint(expr=m.dx == m.vx * u.convert(m.t, to_units=u.s))
        m.dy_con = Constraint(expr=m.dy == m.vy * u.convert(m.t, to_units=u.s) + 0.5 * u.convert(m.a, to_units=u.m / u.s ** 2) * u.convert(m.t, to_units=u.s) ** 2)
        m.ground = Constraint(expr=m.dy == 0)
        m.unitless_con = Constraint(expr=m.x_unitless == 5.0)
        assert_units_consistent(m)
        assert_units_consistent(m.dx)
        assert_units_consistent(m.x_unitless)
        assert_units_consistent(m.vx_con)
        assert_units_consistent(m.unitless_con)
        assert_units_equivalent(m.dx, m.dy)
        assert_units_equivalent(m.x_unitless, u.dimensionless)
        assert_units_equivalent(m.x_unitless, None)
        assert_units_equivalent(m.vx_con.body, u.m / u.s)
        assert_units_equivalent(m.unitless_con.body, u.dimensionless)
        assert_units_equivalent(m.dx, m.dy)
        assert_units_equivalent(m.x_unitless, u.dimensionless)
        assert_units_equivalent(m.x_unitless, None)
        assert_units_equivalent(m.vx_con.body, u.m / u.s)
        m.broken = Constraint(expr=m.dy == 42.0 * u.kg)
        with self.assertRaises(UnitsError):
            assert_units_consistent(m)
        assert_units_consistent(m.dx)
        assert_units_consistent(m.vx_con)
        with self.assertRaises(UnitsError):
            assert_units_consistent(m.broken)
        self.assertTrue(check_units_equivalent(m.dx, m.dy))
        self.assertFalse(check_units_equivalent(m.dx, m.vx))

    def test_assert_units_consistent_on_datas(self):
        u = units
        m = ConcreteModel()
        m.S = Set(initialize=[1, 2, 3])
        m.x = Var(m.S, units=u.m)
        m.t = Var(m.S, units=u.s)
        m.v = Var(m.S, units=u.m / u.s)
        m.unitless = Var(m.S)

        @m.Constraint(m.S)
        def vel_con(m, i):
            return m.v[i] == m.x[i] / m.t[i]

        @m.Constraint(m.S)
        def unitless_con(m, i):
            return m.unitless[i] == 42.0

        @m.Constraint(m.S)
        def sqrt_con(m, i):
            return sqrt(m.v[i]) == sqrt(m.x[i] / m.t[i])
        assert_units_consistent(m)
        assert_units_consistent(m.x)
        assert_units_consistent(m.t)
        assert_units_consistent(m.v)
        assert_units_consistent(m.unitless)
        assert_units_consistent(m.vel_con)
        assert_units_consistent(m.unitless_con)
        assert_units_consistent(m.x[2])
        assert_units_consistent(m.t[2])
        assert_units_consistent(m.v[2])
        assert_units_consistent(m.unitless[2])
        assert_units_consistent(m.vel_con[2])
        assert_units_consistent(m.unitless_con[2])
        assert_units_equivalent(m.x[2], m.x[1])
        assert_units_equivalent(m.t[2], u.s)
        assert_units_equivalent(m.v[2], u.m / u.s)
        assert_units_equivalent(m.unitless[2], u.dimensionless)
        assert_units_equivalent(m.unitless[2], None)
        assert_units_equivalent(m.vel_con[2].body, u.m / u.s)
        assert_units_equivalent(m.unitless_con[2].body, u.dimensionless)

        @m.Constraint(m.S)
        def broken(m, i):
            return m.x[i] == 42.0 * m.v[i]
        with self.assertRaises(UnitsError):
            assert_units_consistent(m)
        with self.assertRaises(UnitsError):
            assert_units_consistent(m.broken)
        with self.assertRaises(UnitsError):
            assert_units_consistent(m.broken[1])
        assert_units_consistent(m.x)
        assert_units_consistent(m.t)
        assert_units_consistent(m.v)
        assert_units_consistent(m.unitless)
        assert_units_consistent(m.vel_con)
        assert_units_consistent(m.unitless_con)
        assert_units_consistent(m.x[2])
        assert_units_consistent(m.t[2])
        assert_units_consistent(m.v[2])
        assert_units_consistent(m.unitless[2])
        assert_units_consistent(m.vel_con[2])
        assert_units_consistent(m.unitless_con[2])

    def test_assert_units_consistent_all_components(self):
        u = units
        m = self._create_model_and_vars()
        m.obj = Objective(expr=m.dx / m.t - m.vx)
        m.con = Constraint(expr=m.dx / m.t == m.vx)
        m.exp = Expression(expr=m.dx / m.t - m.vx)
        m.suff = Suffix(direction=Suffix.LOCAL)
        m.rs = RangeSet(5)
        m.disj1 = Disjunct()
        m.disj1.constraint = Constraint(expr=m.dx / m.t <= m.vx)
        m.disj2 = Disjunct()
        m.disj2.constraint = Constraint(expr=m.dx / m.t <= m.vx)
        m.disjn = Disjunction(expr=[m.disj1, m.disj2])
        m.extfn = ExternalFunction(python_callback_function, units=u.m / u.s, arg_units=[u.m, u.s])
        m.conext = Constraint(expr=m.extfn(m.dx, m.t) - m.vx == 0)
        m.cset = ContinuousSet(bounds=(0, 1))
        m.svar = Var(m.cset, units=u.m)
        m.dvar = DerivativeVar(sVar=m.svar, units=u.m / u.s)

        def prt1_rule(m):
            return {'avar': m.dx}

        def prt2_rule(m):
            return {'avar': m.dy}
        m.prt1 = Port(rule=prt1_rule)
        m.prt2 = Port(rule=prt2_rule)

        def arcrule(m):
            return dict(source=m.prt1, destination=m.prt2)
        m.arc = Arc(rule=arcrule)
        assert_units_consistent(m)

    def test_units_roundoff_error(self):
        m = ConcreteModel()
        m.var_1 = Var(initialize=400, units=units.J ** 0.4 * units.kg ** 0.2 * units.W ** 0.6 / units.K / units.m ** 2.2 / units.Pa ** 0.2 / units.s ** 0.8)
        m.var_1.fix()
        m.var_2 = Var(initialize=400, units=units.kg / units.s ** 3 / units.K)
        assert_units_equivalent(m.var_1, m.var_2)

    def test_identify_inconsistent_units(self):
        u = units
        m = ConcreteModel()
        m.S = Set(initialize=[1, 2, 3])
        m.x = Var(units=u.m)
        m.t = Var(units=u.s)
        m.c1 = Constraint(expr=m.x == 10 * u.m)
        m.c2 = Constraint(expr=m.x == m.t)

        @m.Constraint(m.S)
        def c3(blk, i):
            if i == 1:
                return m.t == 10 * u.m
            return m.t == 10 * u.s
        m.e1 = Expression(expr=m.x + 10 * u.m)
        m.e2 = Expression(expr=m.x + m.t)

        @m.Expression(m.S)
        def e3(blk, i):
            if i == 1:
                return m.t + 10 * u.m
            return m.t + 10 * u.s
        m.o1 = Objective(expr=m.x + 10 * u.m)
        m.o2 = Objective(expr=m.x + m.t)

        @m.Objective(m.S)
        def o3(blk, i):
            if i == 1:
                return m.t + 10 * u.m
            return m.t + 10 * u.s
        failures = identify_inconsistent_units(m)
        assert len(failures) == 6
        assert m.c2 in failures
        assert m.c3[1] in failures
        assert m.e2 in failures
        assert m.e3[1] in failures
        assert m.o2 in failures
        assert m.o3[1] in failures