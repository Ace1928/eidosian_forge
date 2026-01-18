import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
class TestSetsInPython3(unittest.TestCase):

    def test_pprint_mixed(self):
        m = ConcreteModel()
        m.Z = Set(initialize=['A', 'C'])
        m.A = Set(m.Z, initialize={'A': [1, 2, 3, 'A']})
        buf = StringIO()
        m.pprint(ostream=buf)
        ref = "2 Set Declarations\n    A : Size=1, Index=Z, Ordered=Insertion\n        Key : Dimen : Domain : Size : Members\n          A :     1 :    Any :    4 : {1, 2, 3, 'A'}\n    Z : Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    2 : {'A', 'C'}\n\n2 Declarations: Z A\n"
        self.assertEqual(ref, buf.getvalue())

    def test_initialize_and_clone_from_dict_keys(self):
        ref = '1 Set Declarations\n    INDEX : Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    3 : {1, 3, 5}\n\n1 Param Declarations\n    p : Size=3, Index=INDEX, Domain=Any, Default=None, Mutable=False\n        Key : Value\n          1 :     2\n          3 :     4\n          5 :     6\n\n2 Declarations: INDEX p\n'
        m = ConcreteModel()
        v = {1: 2, 3: 4, 5: 6}
        m.INDEX = Set(initialize=v.keys())
        m.p = Param(m.INDEX, initialize=v)
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())
        m2 = m.clone()
        buf = StringIO()
        m2.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())
        m3 = copy.deepcopy(m)
        buf = StringIO()
        m3.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())
        m = ConcreteModel()
        v = {1: 2, 3: 4, 5: 6}
        m.INDEX = Set(initialize=v.keys())
        m.p = Param(m.INDEX, initialize=v)
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())
        m2 = m.clone()
        buf = StringIO()
        m2.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())
        m3 = copy.deepcopy(m)
        buf = StringIO()
        m3.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())