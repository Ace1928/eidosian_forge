from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class TestXmlLoad(LoadTests, unittest.TestCase):
    suffix = 'xml'
    skiplist = ['tableD', 'tableT', 'tableU']

    def test_tableXW_nested1(self):
        self.check_skiplist('tableXW_1')
        with capture_output(currdir + 'loadXW.dat'):
            print('load ' + self.filename('XW_nested1') + " query='./bar/table/*' : [A] X W;")
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_nested2(self):
        self.check_skiplist('tableXW_1')
        with capture_output(currdir + 'loadXW.dat'):
            print('load ' + self.filename('XW_nested2') + " query='./bar/table/row' : [A] X W;")
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')