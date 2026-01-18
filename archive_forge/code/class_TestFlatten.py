import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
class TestFlatten(_TestFlattenBase, unittest.TestCase):

    def setUp(self):
        self._orig_flatten = normalize_index.flatten

    def tearDown(self):
        normalize_index.flatten = self._orig_flatten

    def _model1_1d_sets(self):
        m = ConcreteModel()
        m.time = Set(initialize=[1, 2, 3])
        m.space = Set(initialize=[0.0, 0.5, 1.0])
        m.comp = Set(initialize=['a', 'b'])
        m.v0 = Var()
        m.v1 = Var(m.time)
        m.v2 = Var(m.time, m.space)
        m.v3 = Var(m.time, m.space, m.comp)
        m.v_tt = Var(m.time, m.time)
        m.v_tst = Var(m.time, m.space, m.time)

        @m.Block()
        def b(b):

            @b.Block(m.time)
            def b1(b1):
                b1.v0 = Var()
                b1.v1 = Var(m.space)
                b1.v2 = Var(m.space, m.comp)

                @b1.Block(m.space)
                def b_s(b_s):
                    b_s.v0 = Var()
                    b_s.v1 = Var(m.space)
                    b_s.v2 = Var(m.space, m.comp)

            @b.Block(m.time, m.space)
            def b2(b2):
                b2.v0 = Var()
                b2.v1 = Var(m.comp)
                b2.v2 = Var(m.time, m.comp)
        return m

    def test_flatten_m1_along_time_space(self):
        m = self._model1_1d_sets()
        sets = ComponentSet((m.time, m.space))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 6
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {self._hashRef(m.v0)}
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.time:
                ref_data = {self._hashRef(Reference(m.v1)), self._hashRef(Reference(m.b.b1[:].v0))}
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.time and (sets[1] is m.time):
                ref_data = {self._hashRef(Reference(m.v_tt))}
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                ref_data = {self._hashRef(m.v2), self._hashRef(Reference(m.b.b1[:].v1[:])), self._hashRef(Reference(m.b.b2[:, :].v0)), self._hashRef(Reference(m.b.b1[:].b_s[:].v0))}
                ref_data.update((self._hashRef(Reference(m.v3[:, :, j])) for j in m.comp))
                ref_data.update((self._hashRef(Reference(m.b.b1[:].v2[:, j])) for j in m.comp))
                ref_data.update((self._hashRef(Reference(m.b.b2[:, :].v1[j])) for j in m.comp))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 3 and sets[0] is m.time and (sets[1] is m.space) and (sets[2] is m.time):
                ref_data = {self._hashRef(m.v_tst)}
                ref_data.update((self._hashRef(Reference(m.b.b2[:, :].v2[:, j])) for j in m.comp))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 3 and sets[0] is m.time and (sets[1] is m.space) and (sets[2] is m.space):
                ref_data = {self._hashRef(Reference(m.b.b1[:].b_s[:].v1[:]))}
                (ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[:].v2[:, j])) for j in m.comp)),)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m1_empty(self):
        m = self._model1_1d_sets()
        sets = ComponentSet()
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 1
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {self._hashRef(v) for v in m.component_data_objects(Var)}
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m1_along_space(self):
        m = self._model1_1d_sets()
        sets = ComponentSet((m.space,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3
        T = m.time
        TC = m.time * m.comp
        TT = m.time * m.time
        TTC = m.time * m.time * m.comp
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {self._hashRef(m.v0)}
                ref_data.update((self._hashRef(m.v1[t]) for t in T))
                ref_data.update((self._hashRef(m.v_tt[t1, t2]) for t1, t2 in TT))
                ref_data.update((self._hashRef(m.b.b1[t].v0) for t in T))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.space:
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v2[t, :])) for t in T))
                ref_data.update((self._hashRef(Reference(m.v3[t, :, j])) for t, j in TC))
                ref_data.update((self._hashRef(Reference(m.v_tst[t1, :, t2])) for t1, t2 in TT))
                ref_data.update((self._hashRef(Reference(m.b.b1[t].v1[:])) for t in T))
                ref_data.update((self._hashRef(Reference(m.b.b1[t].v2[:, j])) for t, j in TC))
                ref_data.update((self._hashRef(Reference(m.b.b1[t].b_s[:].v0)) for t in T))
                ref_data.update((self._hashRef(Reference(m.b.b2[t, :].v0)) for t in T))
                ref_data.update((self._hashRef(Reference(m.b.b2[t, :].v1[j])) for t, j in TC))
                ref_data.update((self._hashRef(Reference(m.b.b2[t1, :].v2[t2, j])) for t1, t2, j in TTC))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.space and (sets[1] is m.space):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.b.b1[t].b_s[:].v1[:])) for t in T))
                ref_data.update((self._hashRef(Reference(m.b.b1[t].b_s[:].v2[:, j])) for t, j in TC))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m1_along_time(self):
        m = self._model1_1d_sets()
        sets = ComponentSet((m.time,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        S = m.space
        SS = m.space * m.space
        SC = m.space * m.comp
        SSC = m.space * m.space * m.comp
        assert len(sets_list) == 3
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {self._hashRef(Reference(m.v0))}
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.time:
                ref_data = {self._hashRef(Reference(m.v1)), self._hashRef(Reference(m.b.b1[:].v0))}
                ref_data.update((self._hashRef(Reference(m.v2[:, x])) for x in S))
                ref_data.update((self._hashRef(Reference(m.v3[:, x, j])) for x, j in SC))
                ref_data.update((self._hashRef(Reference(m.b.b1[:].v1[x])) for x in S))
                ref_data.update((self._hashRef(Reference(m.b.b1[:].v2[x, j])) for x, j in SC))
                ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[x].v0)) for x in S))
                ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[x1].v1[x2])) for x1, x2 in SS))
                ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[x1].v2[x2, j])) for x1, x2, j in SSC))
                ref_data.update((self._hashRef(Reference(m.b.b2[:, x].v0)) for x in S))
                ref_data.update((self._hashRef(Reference(m.b.b2[:, x].v1[j])) for x, j in SC))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.time and (sets[1] is m.time):
                ref_data = {self._hashRef(Reference(m.v_tt))}
                ref_data.update((self._hashRef(Reference(m.v_tst[:, x, :])) for x in S))
                ref_data.update((self._hashRef(Reference(m.b.b2[:, x].v2[:, j])) for x, j in SC))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def _model2_nd_sets(self):
        m = ConcreteModel()
        normalize_index.flatten = False
        m.d1 = Set(initialize=[1, 2])
        m.d2 = Set(initialize=[('a', 1), ('b', 2)])
        m.dn = Set(initialize=[('c', 3), ('d', 4, 5)], dimen=None)
        m.v_2n = Var(m.d2, m.dn)
        m.v_12 = Var(m.d1, m.d2)
        m.v_212 = Var(m.d2, m.d1, m.d2)
        m.v_12n = Var(m.d1, m.d2, m.dn)
        m.v_1n2n = Var(m.d1, m.dn, m.d2, m.dn)

        @m.Block(m.d1, m.d2, m.dn)
        def b(b, i1, i2, i3):
            b.v0 = Var()
            b.v1 = Var(m.d1)
            b.v2 = Var(m.d2)
            b.vn = Var(m.dn)
        normalize_index.flatten = True
        return m

    def test_flatten_m2_2d(self):
        """
        This test has some issues due to incompatibility between
        slicing and `normalize_index.flatten==False`.
        """
        m = self._model2_nd_sets()
        sets = ComponentSet((m.d2,))
        normalize_index.flatten = False
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        ref1 = Reference(m.v_2n[:, ('c', 3)])
        ref_set = ref1.index_set()._ref
        self.assertNotIn(('a', 1), ref_set)
        self.assertEqual(len(sets_list), len(comps_list))
        self.assertEqual(len(sets_list), 2)
        normalize_index.flatten = True

    def test_flatten_m2_1d(self):
        m = self._model2_nd_sets()
        sets = ComponentSet((m.d1,))
        normalize_index.flatten = False
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3
        D22 = m.d2 * m.d2
        D2N = m.d2 * m.dn
        DN2N = m.dn.cross(m.d2, m.dn)
        D2NN = m.d2.cross(m.dn, m.dn)
        D2N2 = m.d2.cross(m.dn, m.d2)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is m.d1:
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v_12[:, i2])) for i2 in m.d2))
                ref_data.update((self._hashRef(Reference(m.v_212[i2a, :, i2b])) for i2a, i2b in D22))
                ref_data.update((self._hashRef(Reference(m.v_12n[:, i2, i_n])) for i2, i_n in D2N))
                ref_data.update((self._hashRef(Reference(m.v_1n2n[:, i_na, i2, i_nb])) for i_na, i2, i_nb in DN2N))
                ref_data.update((self._hashRef(Reference(m.b[:, i2, i_n].v0)) for i2, i_n in D2N))
                ref_data.update((self._hashRef(Reference(m.b[:, i2a, i_n].v2[i2b])) for i2a, i_n, i2b in D2N2))
                ref_data.update((self._hashRef(Reference(m.b[:, i2, i_na].vn[i_nb])) for i2, i_na, i_nb in D2NN))
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = set()
                ref_data.update((self._hashRef(v) for v in m.v_2n.values()))
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d1 and (sets[1] is m.d1):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.b[:, i2, i_n].v1[:])) for i2, i_n in D2N))
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()
        normalize_index.flatten = True

    def _model3_nd_sets_normalizeflatten(self):
        m = ConcreteModel()
        m.d1 = Set(initialize=[1, 2])
        m.d2 = Set(initialize=[('a', 1), ('b', 2)])
        m.dn = Set(initialize=[('c', 3), ('d', 4, 5)], dimen=None)
        m.v_2n = Var(m.d2, m.dn)
        m.v_12 = Var(m.d1, m.d2)
        m.v_212 = Var(m.d2, m.d1, m.d2)
        m.v_12n = Var(m.d1, m.d2, m.dn)
        m.v_1n2n = Var(m.d1, m.dn, m.d2, m.dn)
        m.b = Block(m.d1, m.d2, m.dn)
        for i1 in m.d1:
            for i2 in m.d2:
                for i_n in m.dn:
                    m.b[i1, i2, i_n].v0 = Var()
                    m.b[i1, i2, i_n].v1 = Var(m.d1)
                    m.b[i1, i2, i_n].v2 = Var(m.d2)
                    m.b[i1, i2, i_n].vn = Var(m.dn)
        return m

    def test_flatten_m3_1d(self):
        m = self._model3_nd_sets_normalizeflatten()
        sets = ComponentSet((m.d1,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is m.d1:
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v_12[:, i2])) for i2 in m.d2))
                ref_data.update((self._hashRef(Reference(m.v_212[i2a, :, i2b])) for i2a in m.d2 for i2b in m.d2))
                ref_data.update((self._hashRef(Reference(m.v_12n[:, i2, i_n])) for i2 in m.d2 for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.v_1n2n[:, i_na, i2, i_nb])) for i_na in m.dn for i2 in m.d2 for i_nb in m.dn))
                ref_data.update((self._hashRef(Reference(m.b[:, i2, i_n].v0)) for i2 in m.d2 for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.b[:, i2a, i_n].v2[i2b])) for i2a in m.d2 for i_n in m.dn for i2b in m.d2))
                ref_data.update((self._hashRef(Reference(m.b[:, i2, i_na].vn[i_nb])) for i2 in m.d2 for i_na in m.dn for i_nb in m.dn))
                assert len(ref_data) == len(comps)
                assert len(ref_data) == 38
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = set()
                ref_data.update((self._hashRef(v) for v in m.v_2n.values()))
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d1 and (sets[1] is m.d1):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.b[:, i2, i_n].v1[:])) for i2 in m.d2 for i_n in m.dn))
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

    def test_flatten_m3_2d(self):
        m = self._model3_nd_sets_normalizeflatten()
        sets = ComponentSet((m.d2,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 2
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is m.d2:
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v_2n[:, :, i_n])) for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.v_12[i1, :, :])) for i1 in m.d1))
                ref_data.update((self._hashRef(Reference(m.v_12n[i1, :, :, i_n])) for i1 in m.d1 for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.v_1n2n[i1, i_na, :, :, i_nb])) for i1 in m.d1 for i_na in m.dn for i_nb in m.dn))
                ref_data.update((self._hashRef(Reference(m.b[i1, :, :, i_n].v0)) for i1 in m.d1 for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.b[i1a, :, :, i_n].v1[i1b])) for i1a in m.d1 for i_n in m.dn for i1b in m.d1))
                ref_data.update((self._hashRef(Reference(m.b[i1, :, :, i_na].vn[i_nb])) for i1 in m.d1 for i_na in m.dn for i_nb in m.dn))
                assert len(ref_data) == len(comps)
                assert len(ref_data) == 36
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d2 and (sets[1] is m.d2):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v_212[:, :, i1, :, :])) for i1 in m.d1))
                ref_data.update((self._hashRef(Reference(m.b[i1, :, :, i_n].v2[:, :])) for i1 in m.d1 for i_n in m.dn))
                assert len(ref_data) == len(comps)
                assert len(ref_data) == 6
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m3_nd(self):
        m = self._model3_nd_sets_normalizeflatten()
        m.del_component(m.v_1n2n)
        sets = ComponentSet((m.dn,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = set()
                ref_data.update((self._hashRef(v) for v in m.v_12.values()))
                ref_data.update((self._hashRef(v) for v in m.v_212.values()))
                assert len(comps) == len(ref_data)
                assert len(comps) == 12
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.dn:
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v_2n[i2, ...])) for i2 in m.d2))
                ref_data.update((self._hashRef(Reference(m.v_12n[i1, i2, ...])) for i1 in m.d1 for i2 in m.d2))
                ref_data.update((self._hashRef(Reference(m.b[i1, i2, ...].v0)) for i1 in m.d1 for i2 in m.d2))
                ref_data.update((self._hashRef(Reference(m.b[i1a, i2, ...].v1[i1b])) for i1a in m.d1 for i2 in m.d2 for i1b in m.d1))
                ref_data.update((self._hashRef(Reference(m.b[i1, i2a, ...].v2[i2b])) for i1 in m.d1 for i2a in m.d2 for i2b in m.d2))
                assert len(comps) == len(ref_data)
                assert len(comps) == 26
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.dn and (sets[1] is m.dn):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.b[i1, i2, ...].vn[...])) for i1 in m.d1 for i2 in m.d2))
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m3_1_2(self):
        m = self._model3_nd_sets_normalizeflatten()
        sets = ComponentSet((m.d1, m.d2))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 5
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is m.d2:
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v_2n[:, :, i_n])) for i_n in m.dn))
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d1 and (sets[1] is m.d2):
                ref_data = {self._hashRef(Reference(m.v_12[...]))}
                ref_data.update((self._hashRef(Reference(m.v_12n[:, :, :, i_n])) for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.v_1n2n[:, i_na, :, :, i_nb])) for i_na in m.dn for i_nb in m.dn))
                ref_data.update((self._hashRef(Reference(m.b[:, :, :, i_n].v0)) for i_n in m.dn))
                ref_data.update((self._hashRef(Reference(m.b[:, :, :, i_na].vn[i_nb])) for i_na in m.dn for i_nb in m.dn))
                self.assertEqual(len(ref_data), len(comps))
                self.assertEqual(len(comps), 13)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 3 and sets[0] is m.d1 and (sets[1] is m.d2) and (sets[2] is m.d1):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.b[:, :, :, i_n].v1[:])) for i_n in m.dn))
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 3 and sets[0] is m.d1 and (sets[1] is m.d2) and (sets[2] is m.d2):
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.b[:, :, :, i_n].v2[:, :])) for i_n in m.dn))
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 3 and sets[0] is m.d2 and (sets[1] is m.d1) and (sets[2] is m.d2):
                ref_data = {self._hashRef(Reference(m.v_212[...]))}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_specified_index_1(self):
        """
        Components indexed by flattened sets and others
        """
        m = ConcreteModel()
        m.time = Set(initialize=[1, 2, 3])
        m.space = Set(initialize=[2, 4, 6])
        m.phase = Set(initialize=['p1', 'p2'])
        m.comp = Set(initialize=['a', 'b'])
        phase_comp = m.comp * m.phase
        n_phase_comp = len(m.phase) * len(m.comp)
        m.v = Var(m.time, m.comp, m.space, m.phase)

        @m.Block(m.time, m.comp, m.space, m.phase)
        def b(b, t, j, x, p):
            b.v1 = Var()
            if x != 2:
                b.v2 = Var()
        sets = (m.time, m.space)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                self.assertEqual(len(comps), 2 * n_phase_comp)
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v[:, j, :, p])) for j, p in phase_comp))
                ref_data.update((self._hashRef(Reference(m.b[:, j, :, p].v1)) for j, p in phase_comp))
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()
        indices = ComponentMap([(m.space, 4)])
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                self.assertEqual(len(comps), 3 * n_phase_comp)
                incomplete_slices = list((m.b[:, j, :, p].v2 for j, p in phase_comp))
                for ref in incomplete_slices:
                    ref.attribute_errors_generate_exceptions = False
                incomplete_refs = list((Reference(sl) for sl in incomplete_slices))
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v[:, j, :, p])) for j, p in phase_comp))
                ref_data.update((self._hashRef(Reference(m.b[:, j, :, p].v1)) for j, p in phase_comp))
                ref_data.update((self._hashRef(ref) for ref in incomplete_refs))
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()
        indices = (3, 6)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                self.assertEqual(len(comps), 3 * n_phase_comp)
                incomplete_slices = list((m.b[:, j, :, p].v2 for j, p in phase_comp))
                for ref in incomplete_slices:
                    ref.attribute_errors_generate_exceptions = False
                incomplete_refs = list((Reference(sl) for sl in incomplete_slices))
                ref_data = set()
                ref_data.update((self._hashRef(Reference(m.v[:, j, :, p])) for j, p in phase_comp))
                ref_data.update((self._hashRef(Reference(m.b[:, j, :, p].v1)) for j, p in phase_comp))
                ref_data.update((self._hashRef(ref) for ref in incomplete_refs))
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_specified_index_2(self):
        """
        Components indexed only by flattened sets
        """
        m = ConcreteModel()
        m.time = Set(initialize=[1, 2, 3])
        m.space = Set(initialize=[2, 4, 6])
        m.v = Var(m.time, m.space)

        @m.Block(m.time, m.space)
        def b(b, t, x):
            b.v1 = Var()
            if x != 2:
                b.v2 = Var()
        sets = (m.time, m.space)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                self.assertEqual(len(comps), 2)
                ref_data = {self._hashRef(Reference(m.v[...])), self._hashRef(Reference(m.b[...].v1))}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()
        indices = ComponentMap([(m.space, 4)])
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                self.assertEqual(len(comps), 3)
                incomplete_slice = m.b[:, :].v2
                incomplete_slice.attribute_errors_generate_exceptions = False
                incomplete_ref = Reference(incomplete_slice)
                ref_data = {self._hashRef(Reference(m.v[:, :])), self._hashRef(Reference(m.b[:, :].v1)), self._hashRef(incomplete_ref)}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()
        indices = (3, 6)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
                self.assertEqual(len(comps), 3)
                incomplete_slice = m.b[:, :].v2
                incomplete_slice.attribute_errors_generate_exceptions = False
                incomplete_ref = Reference(incomplete_slice)
                ref_data = {self._hashRef(Reference(m.v[:, :])), self._hashRef(Reference(m.b[:, :].v1)), self._hashRef(incomplete_ref)}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def _model4_three_1d_sets(self):
        m = ConcreteModel()
        m.X = Set(initialize=[1, 2, 3])
        m.Y = Set(initialize=[1, 2, 3])
        m.Z = Set(initialize=[1, 2, 3])
        m.comp = Set(initialize=['a', 'b'])
        m.u = Var()
        m.v = Var(m.X, m.Y, m.Z, m.comp)
        m.base = Var(m.X, m.Y)

        @m.Block(m.X, m.Y, m.Z, m.comp)
        def b4(b, x, y, z, j):
            b.v = Var()

        @m.Block(m.X, m.Y)
        def b2(b, x, y):
            b.base = Var()
            b.v = Var(m.Z, m.comp)
        return m

    def test_model4_xyz(self):
        m = self._model4_three_1d_sets()
        sets = (m.X, m.Y, m.Z)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 3)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {self._hashRef(Reference(m.u))}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.X and (sets[1] is m.Y):
                ref_data = {self._hashRef(Reference(m.base[:, :])), self._hashRef(Reference(m.b2[:, :].base))}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 3 and sets[0] is m.X and (sets[1] is m.Y) and (sets[2] is m.Z):
                ref_data = {self._hashRef(Reference(m.v[:, :, :, 'a'])), self._hashRef(Reference(m.v[:, :, :, 'b'])), self._hashRef(Reference(m.b4[:, :, :, 'a'].v)), self._hashRef(Reference(m.b4[:, :, :, 'b'].v)), self._hashRef(Reference(m.b2[:, :].v[:, 'a'])), self._hashRef(Reference(m.b2[:, :].v[:, 'b']))}
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_deactivated_block_active_true(self):
        m = self._model1_1d_sets()
        m.b.b1.deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
        expected_unindexed = [ComponentUID(m.v0)]
        expected_unindexed = set(expected_unindexed)
        expected_time = [ComponentUID(m.v1[:])]
        expected_time.extend((ComponentUID(m.v2[:, x]) for x in m.space))
        expected_time.extend((ComponentUID(m.v3[:, x, j]) for x in m.space for j in m.comp))
        expected_time.extend((ComponentUID(m.b.b2[:, x].v0) for x in m.space))
        expected_time.extend((ComponentUID(m.b.b2[:, x].v1[j]) for x in m.space for j in m.comp))
        expected_time = set(expected_time)
        expected_2time = [ComponentUID(m.v_tt[:, :])]
        expected_2time.extend((ComponentUID(m.v_tst[:, x, :]) for x in m.space))
        expected_2time.extend((ComponentUID(m.b.b2[:, x].v2[:, j]) for x in m.space for j in m.comp))
        expected_2time = set(expected_2time)
        set_id_set = set((tuple((id(s) for s in sets)) for sets in sets_list))
        pred_sets = [(UnindexedComponent_set,), (m.time,), (m.time, m.time)]
        pred_set_ids = set((tuple((id(s) for s in sets)) for sets in pred_sets))
        self.assertEqual(set_id_set, pred_set_ids)
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                comp_set = set((ComponentUID(comp) for comp in comps))
                self.assertEqual(comp_set, expected_unindexed)
            elif len(sets) == 1 and sets[0] is m.time:
                comp_set = set((ComponentUID(comp.referent) for comp in comps))
                self.assertEqual(comp_set, expected_time)
            elif len(sets) == 2:
                self.assertIs(sets[0], m.time)
                self.assertIs(sets[1], m.time)
                comp_set = set((ComponentUID(comp.referent) for comp in comps))
                self.assertEqual(comp_set, expected_2time)

    def test_deactivated_block_active_false(self):
        m = self._model1_1d_sets()
        m.deactivate()
        m.b.deactivate()
        m.b.b1.deactivate()
        m.b.b1[:].b_s.deactivate()
        m.del_component(m.v0)
        m.del_component(m.v1)
        m.del_component(m.v3)
        m.del_component(m.v_tt)
        m.del_component(m.v_tst)
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=False)
        expected_time = [ComponentUID(m.b.b1[:].v0)]
        expected_time.extend((ComponentUID(m.v2[:, x]) for x in m.space))
        expected_time.extend((ComponentUID(m.b.b1[:].v1[x]) for x in m.space))
        expected_time.extend((ComponentUID(m.b.b1[:].v2[x, j]) for x in m.space for j in m.comp))
        expected_time.extend((ComponentUID(m.b.b1[:].b_s[x].v0) for x in m.space))
        expected_time.extend((ComponentUID(m.b.b1[:].b_s[x1].v1[x2]) for x1 in m.space for x2 in m.space))
        expected_time.extend((ComponentUID(m.b.b1[:].b_s[x1].v2[x2, j]) for x1 in m.space for x2 in m.space for j in m.comp))
        expected_time = set(expected_time)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        comp_set = set((ComponentUID(comp.referent) for comp in comps_list[0]))
        self.assertEqual(comp_set, expected_time)

    def test_partially_deactivated_slice_active_true(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        for t in m.time:
            m.b[t].v = Var()
        m.b[0].deactivate()
        m.b[1].deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.b[:].v))

    def test_partially_activated_slice_active_false(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        m.deactivate()
        m.b.deactivate()
        for t in m.time:
            m.b[t].v = Var()
        m.b[0].deactivate()
        m.b[1].deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=False)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.b[:].v))

    def test_partially_deactivated_slice_specified_index(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        for t in m.time:
            m.b[t].v = Var()
        m.b[0].deactivate()
        m.b[1].deactivate()
        sets = (m.time,)
        indices = (1,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True, indices=indices)
        self.assertEqual(len(sets_list), 0)
        self.assertEqual(len(comps_list), 0)
        indices = (2,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True, indices=indices)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.b[:].v))

    def test_fully_deactivated_slice(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        for t in m.time:
            m.b[t].v = Var()
        m.b[:].deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
        self.assertEqual(len(sets_list), 0)
        self.assertEqual(len(comps_list), 0)

    def test_deactivated_model_active_false(self):
        m = self._model1_1d_sets()
        m.deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
        self.assertEqual(len(sets_list), 0)
        self.assertEqual(len(comps_list), 0)

    def test_constraint_with_active_arg(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        for t in m.time:
            m.b[t].v = Var()
            m.b[t].c1 = Constraint(expr=m.b[t].v == 1)

        def c2_rule(m, t):
            return m.b[t].v == 2
        m.c2 = Constraint(m.time, rule=c2_rule)
        m.c2.deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.b[:].c1))
        m.deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=False)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.c2[:]))

    def test_constraint_partially_deactivated_slice(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        for t in m.time:
            m.b[t].v = Var()

        def c2_rule(m, t):
            return m.b[t].v == 2
        m.c2 = Constraint(m.time, rule=c2_rule)
        m.c2[0].deactivate()
        m.c2[1].deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], m.time)
        self.assertIs(sets_list[0][0], m.time)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.c2[:]))

    def test_constraint_fully_deactivated_slice(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2, 3])
        m.b = Block(m.time)
        for t in m.time:
            m.b[t].v = Var()

        def c2_rule(m, t):
            return m.b[t].v == 2
        m.c2 = Constraint(m.time, rule=c2_rule)
        m.c2[:].deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
        self.assertEqual(len(sets_list), 0)
        self.assertEqual(len(comps_list), 0)

    def test_scalar_con_active_true(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2])
        m.v = Var()
        m.c = Constraint(expr=m.v == 1)
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
        self.assertEqual(len(sets_list), 1)
        self.assertEqual(len(sets_list[0]), 1)
        self.assertIs(sets_list[0][0], UnindexedComponent_set)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), 1)
        self.assertIs(comps_list[0][0], m.c)

    def test_deactivated_scalar_con_active_true(self):
        m = ConcreteModel()
        m.time = Set(initialize=[0, 1, 2])
        m.comp = Set(initialize=['A', 'B'])
        m.v = Var()

        def c_rule(m, j):
            return m.v == 1
        m.c = Constraint(m.comp, rule=c_rule)
        m.c[:].deactivate()
        sets = (m.time,)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
        self.assertEqual(len(sets_list), 0)
        self.assertEqual(len(comps_list), 0)