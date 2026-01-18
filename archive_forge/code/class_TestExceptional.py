import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
class TestExceptional(unittest.TestCase):
    """
    These are the cases that motivate the try/excepts in the slice-checking
    part of the code.
    """

    def test_stop_iteration(self):
        """
        StopIteration is raised if we create an empty slice somewhere
        along the line. It is an open question what we should do in the
        case of an empty slice, but my preference is to omit it so we
        don't return a reference that doesn't admit any valid indices.
        """
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=['a', 'b', 'c'])
        m.v = Var(m.s1, m.s2)

        def con_rule(m, i, j):
            if j == 'a':
                return Constraint.Skip
            return m.v[i, j] == 5.0

        def vacuous_con_rule(m, i, j):
            return Constraint.Skip
        m.con = Constraint(m.s1, m.s2, rule=con_rule)
        with self.assertRaises(StopIteration):
            next(iter(m.con[:, 'a']))
        sets = (m.s1,)
        ctype = Constraint
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), len(m.s2) - 1)
        m.del_component(m.con)
        m.vacuous_con = Constraint(m.s1, m.s2, rule=vacuous_con_rule)
        with self.assertRaises(StopIteration):
            next(iter(m.vacuous_con[...]))
        sets = (m.s1, m.s2)
        ctype = Constraint
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 0)
        m.del_component(m.vacuous_con)
        m.del_component(m.v)

        def block_rule(b, i, j):
            b.v = Var()
        m.b = Block(m.s1, m.s2, rule=block_rule)
        for i in m.s1:
            del m.b[i, 'a']
        with self.assertRaises(StopIteration):
            next(iter(m.b[:, 'a'].v))
        sets = (m.s1,)
        ctype = Var
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), len(m.s2) - 1)
        for idx in m.b:
            del m.b[idx]
        with self.assertRaises(StopIteration):
            next(iter(m.b[...].v))
        sets = (m.s1, m.s2)
        ctype = Var
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 0)
        subset_set = ComponentSet(m.b.index_set().subsets())
        for s in sets:
            self.assertIn(s, subset_set)

    def test_descend_stop_iteration(self):
        """
        Even if we construct a non-empty slice, if we provide a bad
        index to descend into, we can end up with no valid blocks
        to descend into. Unclear whether we should raise an error here.
        """
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=['a', 'b'])
        m.v = Var(m.s1, m.s2)

        def b_rule(b, i, j):
            b.v = Var()
        m.b = Block(m.s1, m.s2, rule=b_rule)
        for i in m.s1:
            del m.b[i, 'b']
        with self.assertRaises(StopIteration):
            next(iter(m.b[:, 'b']))
        sets = (m.s1, m.s2)
        ctype = Var
        indices = ComponentMap([(m.s2, 'b')])
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype, indices=indices)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.s1 and (sets[1] is m.s2):
                self.assertEqual(len(comps), 1)
                self.assertEqual(str(ComponentUID(comps[0].referent)), 'v[*,*]')
            else:
                raise RuntimeError()

    def test_bad_descend_index(self):
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=['a', 'b'])
        m.v = Var(m.s1, m.s2)

        def b_rule(b, i, j):
            b.v = Var()
        m.b = Block(m.s1, m.s2, rule=b_rule)
        sets = (m.s1, m.s2)
        ctype = Var
        indices = ComponentMap([(m.s1, 'b')])
        with self.assertRaisesRegex(ValueError, 'bad index'):
            sets_list, comps_list = flatten_components_along_sets(m, sets, ctype, indices=indices)

    def test_keyerror(self):
        """
        KeyErrors occur when a component that we don't slice
        doesn't have data for some members of its indexing set.
        """
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=['a', 'b', 'c'])
        m.v = Var(m.s1, m.s2)

        def con_rule(m, i, j):
            if j == 'a':
                return Constraint.Skip
            return m.v[i, j] == 5.0
        m.con = Constraint(m.s1, m.s2, rule=con_rule)
        with self.assertRaises(KeyError):
            for idx in m.con.index_set():
                temp = m.con[idx]
        sets = ()
        ctype = Constraint
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(sets_list), len(comps_list))
        self.assertEqual(len(sets_list), 1)
        self.assertIs(sets_list[0][0], UnindexedComponent_set)
        self.assertEqual(len(comps_list[0]), len(list(m.con.values())))