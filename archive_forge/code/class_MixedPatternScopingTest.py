from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
class MixedPatternScopingTest(test.TestCase):

    def test_graph_linear_scope(self):
        r = gf.Flow('root')
        r_1 = test_utils.TaskOneReturn('root.1')
        r_2 = test_utils.TaskOneReturn('root.2')
        r.add(r_1, r_2)
        r.link(r_1, r_2)
        s = lf.Flow('subroot')
        s_1 = test_utils.TaskOneReturn('subroot.1')
        s_2 = test_utils.TaskOneReturn('subroot.2')
        s.add(s_1, s_2)
        r.add(s)
        t = gf.Flow('subroot2')
        t_1 = test_utils.TaskOneReturn('subroot2.1')
        t_2 = test_utils.TaskOneReturn('subroot2.2')
        t.add(t_1, t_2)
        t.link(t_1, t_2)
        r.add(t)
        r.link(s, t)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual([], _get_scopes(c, r_1))
        self.assertEqual([['root.1']], _get_scopes(c, r_2))
        self.assertEqual([], _get_scopes(c, s_1))
        self.assertEqual([['subroot.1']], _get_scopes(c, s_2))
        self.assertEqual([[], ['subroot.2', 'subroot.1']], _get_scopes(c, t_1))
        self.assertEqual([['subroot2.1'], ['subroot.2', 'subroot.1']], _get_scopes(c, t_2))

    def test_linear_unordered_scope(self):
        r = lf.Flow('root')
        r_1 = test_utils.TaskOneReturn('root.1')
        r_2 = test_utils.TaskOneReturn('root.2')
        r.add(r_1, r_2)
        u = uf.Flow('subroot')
        atoms = []
        for i in range(0, 5):
            atoms.append(test_utils.TaskOneReturn('subroot.%s' % i))
        u.add(*atoms)
        r.add(u)
        r_3 = test_utils.TaskOneReturn('root.3')
        r.add(r_3)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual([], _get_scopes(c, r_1))
        self.assertEqual([['root.1']], _get_scopes(c, r_2))
        for a in atoms:
            self.assertEqual([[], ['root.2', 'root.1']], _get_scopes(c, a))
        scope = _get_scopes(c, r_3)
        self.assertEqual(1, len(scope))
        first_root = 0
        for i, n in enumerate(scope[0]):
            if n.startswith('root.'):
                first_root = i
                break
        first_subroot = 0
        for i, n in enumerate(scope[0]):
            if n.startswith('subroot.'):
                first_subroot = i
                break
        self.assertGreater(first_subroot, first_root)
        self.assertEqual(['root.2', 'root.1'], scope[0][-2:])

    def test_shadow_graph(self):
        r = gf.Flow('root')
        customer = test_utils.ProvidesRequiresTask('customer', provides=['dog'], requires=[])
        customer2 = test_utils.ProvidesRequiresTask('customer2', provides=['dog'], requires=[])
        washer = test_utils.ProvidesRequiresTask('washer', requires=['dog'], provides=['wash'])
        r.add(customer, washer)
        r.add(customer2, resolve_requires=False)
        r.link(customer2, washer)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual(set(['customer', 'customer2']), set(_get_scopes(c, washer)[0]))
        self.assertEqual([], _get_scopes(c, customer2))
        self.assertEqual([], _get_scopes(c, customer))

    def test_shadow_linear(self):
        r = lf.Flow('root')
        customer = test_utils.ProvidesRequiresTask('customer', provides=['dog'], requires=[])
        customer2 = test_utils.ProvidesRequiresTask('customer2', provides=['dog'], requires=[])
        washer = test_utils.ProvidesRequiresTask('washer', requires=['dog'], provides=['wash'])
        r.add(customer, customer2, washer)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual(['customer2', 'customer'], _get_scopes(c, washer)[0])