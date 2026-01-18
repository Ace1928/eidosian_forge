from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class LoadTests(object):
    skiplist = []

    def check_skiplist(self, name):
        self.skipTest('Skipping load tests')

    def Xcheck_skiplist(self, name):
        if name in self.skiplist:
            self.skipTest('Skipping test %s' % name)

    def filename(self, tname):
        return os.path.abspath(tutorial_dir + os.sep + self.suffix + os.sep + tname + '.' + self.suffix)

    def test_tableA1(self):
        self.check_skiplist('tableA1')
        with capture_output(currdir + 'loadA1.dat'):
            print('load ' + self.filename('A') + ' format=set : A;')
        model = AbstractModel()
        model.A = Set()
        instance = model.create_instance(currdir + 'loadA1.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        os.remove(currdir + 'loadA1.dat')

    def test_tableA2(self):
        self.check_skiplist('tableA2')
        with capture_output(currdir + 'loadA2.dat'):
            print('load ' + self.filename('A') + ' ;')
        model = AbstractModel()
        model.A = Set()
        try:
            instance = model.create_instance(currdir + 'loadA2.dat')
            self.fail('Should fail because no set name is specified')
        except IOError:
            pass
        except IndexError:
            pass
        os.remove(currdir + 'loadA2.dat')

    def test_tableA3(self):
        self.check_skiplist('tableA3')
        with capture_output(currdir + 'loadA3.dat'):
            print('load ' + self.filename('A') + ' format=set : A ;')
        model = AbstractModel()
        model.A = Set()
        instance = model.create_instance(currdir + 'loadA3.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        os.remove(currdir + 'loadA3.dat')

    def test_tableB1(self):
        self.check_skiplist('tableB1')
        with capture_output(currdir + 'loadB.dat'):
            print('load ' + self.filename('B') + ' format=set : B;')
        model = AbstractModel()
        model.B = Set()
        instance = model.create_instance(currdir + 'loadB.dat')
        self.assertEqual(set(instance.B.data()), set([1, 2, 3]))
        os.remove(currdir + 'loadB.dat')

    def test_tableC(self):
        self.check_skiplist('tableC')
        with capture_output(currdir + 'loadC.dat'):
            print('load ' + self.filename('C') + ' format=set : C ;')
        model = AbstractModel()
        model.C = Set(dimen=2)
        instance = model.create_instance(currdir + 'loadC.dat')
        self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A1', 2), ('A1', 3), ('A2', 1), ('A2', 2), ('A2', 3), ('A3', 1), ('A3', 2), ('A3', 3)]))
        os.remove(currdir + 'loadC.dat')

    def test_tableD(self):
        self.check_skiplist('tableD')
        with capture_output(currdir + 'loadD.dat'):
            print('load ' + self.filename('D') + ' format=set_array : C ;')
        model = AbstractModel()
        model.C = Set(dimen=2)
        instance = model.create_instance(currdir + 'loadD.dat')
        self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A2', 2), ('A3', 3)]))
        os.remove(currdir + 'loadD.dat')

    def test_tableZ(self):
        self.check_skiplist('tableZ')
        with capture_output(currdir + 'loadZ.dat'):
            print('load ' + self.filename('Z') + ' : Z ;')
        model = AbstractModel()
        model.Z = Param(default=99.0)
        instance = model.create_instance(currdir + 'loadZ.dat')
        self.assertEqual(instance.Z, 1.01)
        os.remove(currdir + 'loadZ.dat')

    def test_tableY(self):
        self.check_skiplist('tableY')
        with capture_output(currdir + 'loadY.dat'):
            print('load ' + self.filename('Y') + ' : [A] Y;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.Y = Param(model.A)
        instance = model.create_instance(currdir + 'loadY.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.Y.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        os.remove(currdir + 'loadY.dat')

    def test_tableXW_1(self):
        self.check_skiplist('tableXW_1')
        with capture_output(currdir + 'loadXW.dat'):
            print('load ' + self.filename('XW') + ' : [A] X W;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_2(self):
        self.check_skiplist('tableXW_2')
        with capture_output(currdir + 'loadXW.dat'):
            print('load ' + self.filename('XW') + ' : [A] X W;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_3(self):
        self.check_skiplist('tableXW_3')
        with capture_output(currdir + 'loadXW.dat'):
            print('load ' + self.filename('XW') + ' : A=[A] X W;')
        model = AbstractModel()
        model.A = Set()
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_4(self):
        self.check_skiplist('tableXW_4')
        with capture_output(currdir + 'loadXW.dat'):
            print('load ' + self.filename('XW') + ' : B=[A] R=X S=W;')
        model = AbstractModel()
        model.B = Set()
        model.R = Param(model.B)
        model.S = Param(model.B)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.B.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.R.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.S.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableT(self):
        self.check_skiplist('tableT')
        with capture_output(currdir + 'loadT.dat'):
            print('load ' + self.filename('T') + ' format=transposed_array : T;')
        model = AbstractModel()
        model.B = Set(initialize=['I1', 'I2', 'I3', 'I4'])
        model.A = Set(initialize=['A1', 'A2', 'A3'])
        model.T = Param(model.A, model.B)
        instance = model.create_instance(currdir + 'loadT.dat')
        self.assertEqual(instance.T.extract_values(), {('A2', 'I1'): 2.3, ('A1', 'I2'): 1.4, ('A1', 'I3'): 1.5, ('A1', 'I4'): 1.6, ('A1', 'I1'): 1.3, ('A3', 'I4'): 3.6, ('A2', 'I4'): 2.6, ('A3', 'I1'): 3.3, ('A2', 'I3'): 2.5, ('A3', 'I2'): 3.4, ('A2', 'I2'): 2.4, ('A3', 'I3'): 3.5})
        os.remove(currdir + 'loadT.dat')

    def test_tableU(self):
        self.check_skiplist('tableU')
        with capture_output(currdir + 'loadU.dat'):
            print('load ' + self.filename('U') + ' format=array : U;')
        model = AbstractModel()
        model.A = Set(initialize=['I1', 'I2', 'I3', 'I4'])
        model.B = Set(initialize=['A1', 'A2', 'A3'])
        model.U = Param(model.A, model.B)
        instance = model.create_instance(currdir + 'loadU.dat')
        self.assertEqual(instance.U.extract_values(), {('I2', 'A1'): 1.4, ('I3', 'A1'): 1.5, ('I3', 'A2'): 2.5, ('I4', 'A1'): 1.6, ('I3', 'A3'): 3.5, ('I1', 'A2'): 2.3, ('I4', 'A3'): 3.6, ('I1', 'A3'): 3.3, ('I4', 'A2'): 2.6, ('I2', 'A3'): 3.4, ('I1', 'A1'): 1.3, ('I2', 'A2'): 2.4})
        os.remove(currdir + 'loadU.dat')

    def test_tableS(self):
        self.check_skiplist('tableS')
        with capture_output(currdir + 'loadS.dat'):
            print('load ' + self.filename('S') + ' : [A] S ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.S = Param(model.A)
        instance = model.create_instance(currdir + 'loadS.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.S.extract_values(), {'A1': 3.3, 'A3': 3.5})
        os.remove(currdir + 'loadS.dat')

    def test_tablePO(self):
        self.check_skiplist('tablePO')
        with capture_output(currdir + 'loadPO.dat'):
            print('load ' + self.filename('PO') + ' : J=[A,B] P O;')
        model = AbstractModel()
        model.J = Set(dimen=2)
        model.P = Param(model.J)
        model.O = Param(model.J)
        instance = model.create_instance(currdir + 'loadPO.dat')
        self.assertEqual(set(instance.J.data()), set([('A3', 'B3'), ('A1', 'B1'), ('A2', 'B2')]))
        self.assertEqual(instance.P.extract_values(), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})
        self.assertEqual(instance.O.extract_values(), {('A3', 'B3'): 5.5, ('A1', 'B1'): 5.3, ('A2', 'B2'): 5.4})
        os.remove(currdir + 'loadPO.dat')