from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class TestOnlyTextPortal(unittest.TestCase):
    suffix = '.tab'
    skiplist = []

    def check_skiplist(self, name):
        if name in self.skiplist:
            self.skipTest('Skipping test %s' % name)

    def create_options(self, name):
        return {'filename': os.path.abspath(tutorial_dir + os.sep + 'tab' + os.sep + name + self.suffix)}

    def create_write_options(self, name):
        return {'filename': os.path.abspath(currdir + os.sep + name + self.suffix), 'sort': True}

    def test_empty(self):
        self.check_skiplist('empty')
        dp = DataPortal()
        try:
            dp.load(set='A', filename=os.path.abspath(currdir + os.sep + 'empty' + self.suffix))
            self.fail('Expected IOError')
        except IOError:
            pass

    def test_tableA(self):
        self.check_skiplist('tableA')
        dp = DataPortal()
        dp.load(set='A', **self.create_options('A'))
        self.assertEqual(set(dp.data('A')), set(['A1', 'A2', 'A3']))

    def test_tableB(self):
        self.check_skiplist('tableB')
        dp = DataPortal()
        dp.load(set='B', **self.create_options('B'))
        self.assertEqual(set(dp.data('B')), set([1, 2, 3]))

    def test_tableC(self):
        self.check_skiplist('tableC')
        dp = DataPortal()
        dp.load(set='C', **self.create_options('C'))
        self.assertEqual(set(dp.data('C')), set([('A1', 1), ('A1', 2), ('A1', 3), ('A2', 1), ('A2', 2), ('A2', 3), ('A3', 1), ('A3', 2), ('A3', 3)]))

    def test_tableD(self):
        self.check_skiplist('tableD')
        dp = DataPortal()
        dp.load(set='D', format='set_array', **self.create_options('D'))
        self.assertEqual(set(dp.data('D')), set([('A1', 1), ('A2', 2), ('A3', 3)]))

    def test_tableZ(self):
        self.check_skiplist('tableZ')
        dp = DataPortal()
        dp.load(param='Z', **self.create_options('Z'))
        self.assertEqual(dp.data('Z'), 1.01)

    def test_tableY(self):
        self.check_skiplist('tableY')
        dp = DataPortal()
        dp.load(param='Y', **self.create_options('Y'))
        self.assertEqual(dp.data('Y'), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})

    def test_tableXW_1(self):
        self.check_skiplist('tableXW_1')
        dp = DataPortal()
        dp.load(param=('X', 'W'), **self.create_options('XW'))
        self.assertEqual(dp.data('X'), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(dp.data('W'), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableXW_3(self):
        self.check_skiplist('tableXW_3')
        dp = DataPortal()
        dp.load(index='A', param=('X', 'W'), **self.create_options('XW'))
        self.assertEqual(set(dp.data('A')), set(['A1', 'A2', 'A3']))
        self.assertEqual(dp.data('X'), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(dp.data('W'), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableXW_4(self):
        self.check_skiplist('tableXW_4')
        dp = DataPortal()
        dp.load(select=('A', 'W', 'X'), index='B', param=('R', 'S'), **self.create_options('XW'))
        self.assertEqual(set(dp.data('B')), set(['A1', 'A2', 'A3']))
        self.assertEqual(dp.data('S'), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(dp.data('R'), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableT(self):
        self.check_skiplist('tableT')
        dp = DataPortal()
        dp.load(format='transposed_array', param='T', **self.create_options('T'))
        self.assertEqual(dp.data('T'), {('A2', 'I1'): 2.3, ('A1', 'I2'): 1.4, ('A1', 'I3'): 1.5, ('A1', 'I4'): 1.6, ('A1', 'I1'): 1.3, ('A3', 'I4'): 3.6, ('A2', 'I4'): 2.6, ('A3', 'I1'): 3.3, ('A2', 'I3'): 2.5, ('A3', 'I2'): 3.4, ('A2', 'I2'): 2.4, ('A3', 'I3'): 3.5})

    def test_tableU(self):
        self.check_skiplist('tableU')
        dp = DataPortal()
        dp.load(format='array', param='U', **self.create_options('U'))
        self.assertEqual(dp.data('U'), {('I2', 'A1'): 1.4, ('I3', 'A1'): 1.5, ('I3', 'A2'): 2.5, ('I4', 'A1'): 1.6, ('I3', 'A3'): 3.5, ('I1', 'A2'): 2.3, ('I4', 'A3'): 3.6, ('I1', 'A3'): 3.3, ('I4', 'A2'): 2.6, ('I2', 'A3'): 3.4, ('I1', 'A1'): 1.3, ('I2', 'A2'): 2.4})

    def test_tableS(self):
        self.check_skiplist('tableS')
        dp = DataPortal()
        dp.load(param='S', **self.create_options('S'))
        self.assertEqual(dp.data('S'), {'A1': 3.3, 'A3': 3.5})

    def test_tablePO(self):
        self.check_skiplist('tablePO')
        dp = DataPortal()
        dp.load(index='J', param=('P', 'O'), **self.create_options('PO'))
        self.assertEqual(set(dp.data('J')), set([('A3', 'B3'), ('A1', 'B1'), ('A2', 'B2')]))
        self.assertEqual(dp.data('P'), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})
        self.assertEqual(dp.data('O'), {('A3', 'B3'): 5.5, ('A1', 'B1'): 5.3, ('A2', 'B2'): 5.4})

    def test_tablePP(self):
        self.check_skiplist('tablePP')
        dp = DataPortal()
        dp.load(param='PP', **self.create_options('PP'))
        self.assertEqual(dp.data('PP'), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})