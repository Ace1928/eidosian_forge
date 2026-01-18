import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
class Test_SetOperator(unittest.TestCase):

    def test_construct(self):
        p = Param(initialize=3)
        a = RangeSet(p, name='a')
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i = a * a
            self.assertEqual(output.getvalue(), '')
        p.construct()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i.construct()
            ref = 'Constructing SetOperator, name=a*a, from data=None\nConstructing RangeSet, name=a, from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            i.construct()
            self.assertEqual(output.getvalue(), ref)

    def test_deepcopy(self):
        a = AbstractModel()
        a.A = Set(initialize=[1, 2])
        a.B = Set(initialize=[3, 4])

        def x_init(m, i):
            if i == 2:
                return Set.Skip
            else:
                return []
        a.x = Set([1, 2], domain={1: a.A * a.B, 2: a.A * a.A}, initialize=x_init)
        i = a.create_instance()
        self.assertEqual(len(i.x), 1)
        self.assertIn(1, i.x)
        self.assertNotIn(2, i.x)
        self.assertEqual(i.x[1].dimen, 2)
        self.assertEqual(i.x[1].domain, i.A * i.B)
        self.assertEqual(i.x[1], [])

    @unittest.skipIf(not pandas_available, 'pandas is not available')
    def test_pandas_multiindex_set_init(self):
        iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
        pandas_index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])
        model = ConcreteModel()
        model.a = Set(initialize=pandas_index, dimen=pandas_index.nlevels)
        model.b = Set(initialize=pandas_index)
        self.assertIsInstance(model.a, Set)
        self.assertEqual(list(model.a), list(pandas_index))
        self.assertEqual(model.a.dimen, pandas_index.nlevels)
        self.assertIsInstance(model.b, Set)
        self.assertEqual(list(model.b), list(pandas_index))
        self.assertEqual(model.b.dimen, pandas_index.nlevels)