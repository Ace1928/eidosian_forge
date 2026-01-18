from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
@unittest.skipIf(not xls_interface, 'No XLS interface available')
class PyomoTableData(unittest.TestCase):

    def setUp(self):
        pass

    def construct(self, filename):
        pass

    def test_read_set(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', format='set', set='X')
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['set', 'X', ':=', ('A1', 2.0, 3.0, 4.0), ('A5', 6.0, 7.0, 8.0), ('A9', 10.0, 11.0, 12.0), ('A13', 14.0, 15.0, 16.0)])
        except ApplicationError:
            pass

    def test_read_param1(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', param=['bb', 'cc', 'dd'])
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['param', ':', 'bb', 'cc', 'dd', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
        except ApplicationError:
            pass

    def test_read_param2(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', index='X', param=['bb', 'cc', 'dd'])
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['param', ':', 'X', ':', 'bb', 'cc', 'dd', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
        except ApplicationError:
            pass

    def test_read_param3(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', index='X', param=['a'])
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['param', ':', 'X', ':', 'a', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
        except ApplicationError:
            pass

    def test_read_param4(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', index='X', param=['a', 'b'])
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['param', ':', 'X', ':', 'a', 'b', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
        except ApplicationError:
            pass

    def test_read_array1(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', param='X', format='array')
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['param', 'X', ':', 'bb', 'cc', 'dd', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
        except ApplicationError:
            pass

    def test_read_array2(self):
        td = DataManagerFactory('xls')
        td.initialize(filename=currdir + 'Book1.xls', range='TheRange', param='X', format='transposed_array')
        try:
            td.open()
            td.read()
            td.close()
            self.assertEqual(td._info, ['param', 'X', '(tr)', ':', 'bb', 'cc', 'dd', ':=', 'A1', 2.0, 3.0, 4.0, 'A5', 6.0, 7.0, 8.0, 'A9', 10.0, 11.0, 12.0, 'A13', 14.0, 15.0, 16.0])
        except ApplicationError:
            pass

    def test_error1(self):
        td = DataManagerFactory('xls')
        td.initialize(filename='bad')
        try:
            td.open()
            self.fail('Expected IOError because of bad file')
        except IOError:
            pass

    def test_error2(self):
        td = DataManagerFactory('xls')
        try:
            td.open()
            self.fail('Expected IOError because no file specified')
        except (IOError, AttributeError):
            pass

    def test_error3(self):
        td = DataManagerFactory('txt')
        try:
            td.initialize(filename=currdir + 'display.txt')
            td.open()
            self.fail('Expected IOError because of bad file type')
        except (IOError, AttributeError):
            pass

    def test_error4(self):
        td = DataManagerFactory('txt')
        try:
            td.initialize(filename=currdir + 'dummy')
            td.open()
            self.fail('Expected IOError because of bad file type')
        except (IOError, AttributeError):
            pass

    def test_error5(self):
        td = DataManagerFactory('tab')
        td.initialize(filename=example_dir + 'D.tab', param='D', format='foo')
        td.open()
        try:
            td.read()
            self.fail('Expected IOError because of bad format')
        except ValueError:
            pass