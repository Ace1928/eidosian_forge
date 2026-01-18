from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
class GraphScopingTest(test.TestCase):

    def test_dependent(self):
        r = gf.Flow('root')
        customer = test_utils.ProvidesRequiresTask('customer', provides=['dog'], requires=[])
        washer = test_utils.ProvidesRequiresTask('washer', requires=['dog'], provides=['wash'])
        dryer = test_utils.ProvidesRequiresTask('dryer', requires=['dog', 'wash'], provides=['dry_dog'])
        shaved = test_utils.ProvidesRequiresTask('shaver', requires=['dry_dog'], provides=['shaved_dog'])
        happy_customer = test_utils.ProvidesRequiresTask('happy_customer', requires=['shaved_dog'], provides=['happiness'])
        r.add(customer, washer, dryer, shaved, happy_customer)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual([], _get_scopes(c, customer))
        self.assertEqual([['washer', 'customer']], _get_scopes(c, dryer))
        self.assertEqual([['shaver', 'dryer', 'washer', 'customer']], _get_scopes(c, happy_customer))

    def test_no_visible(self):
        r = gf.Flow('root')
        atoms = []
        for i in range(0, 10):
            atoms.append(test_utils.TaskOneReturn('root.%s' % i))
        r.add(*atoms)
        c = compiler.PatternCompiler(r).compile()
        for a in atoms:
            self.assertEqual([], _get_scopes(c, a))

    def test_nested(self):
        r = gf.Flow('root')
        r_1 = test_utils.TaskOneReturn('root.1')
        r_2 = test_utils.TaskOneReturn('root.2')
        r.add(r_1, r_2)
        r.link(r_1, r_2)
        subroot = gf.Flow('subroot')
        subroot_r_1 = test_utils.TaskOneReturn('subroot.1')
        subroot_r_2 = test_utils.TaskOneReturn('subroot.2')
        subroot.add(subroot_r_1, subroot_r_2)
        subroot.link(subroot_r_1, subroot_r_2)
        r.add(subroot)
        r_3 = test_utils.TaskOneReturn('root.3')
        r.add(r_3)
        r.link(r_2, r_3)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual([], _get_scopes(c, r_1))
        self.assertEqual([['root.1']], _get_scopes(c, r_2))
        self.assertEqual([['root.2', 'root.1']], _get_scopes(c, r_3))
        self.assertEqual([], _get_scopes(c, subroot_r_1))
        self.assertEqual([['subroot.1']], _get_scopes(c, subroot_r_2))