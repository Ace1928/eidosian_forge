from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class TestTableCmd(unittest.TestCase):

    def test_tableA1_1(self):
        with capture_output(currdir + 'loadA1.dat'):
            print('table columns=1 A={1} := A1 A2 A3 ;')
        model = AbstractModel()
        model.A = Set()
        instance = model.create_instance(currdir + 'loadA1.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        os.remove(currdir + 'loadA1.dat')

    def test_tableA1_2(self):
        with capture_output(currdir + 'loadA1.dat'):
            print('table A={A} : A := A1 A2 A3 ;')
        model = AbstractModel()
        model.A = Set()
        instance = model.create_instance(currdir + 'loadA1.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        os.remove(currdir + 'loadA1.dat')

    def test_tableB1_1(self):
        with capture_output(currdir + 'loadB.dat'):
            print('table columns=1 B={1} := 1 2 3 ;')
        model = AbstractModel()
        model.B = Set()
        instance = model.create_instance(currdir + 'loadB.dat')
        self.assertEqual(set(instance.B.data()), set([1, 2, 3]))
        os.remove(currdir + 'loadB.dat')

    def test_tableB1_2(self):
        with capture_output(currdir + 'loadB.dat'):
            print('table B={B} : B := 1 2 3 ;')
        model = AbstractModel()
        model.B = Set()
        instance = model.create_instance(currdir + 'loadB.dat')
        self.assertEqual(set(instance.B.data()), set([1, 2, 3]))
        os.remove(currdir + 'loadB.dat')

    def test_tableC_1(self):
        with capture_output(currdir + 'loadC.dat'):
            print('table columns=2 C={1,2} := A1 1 A1 2 A1 3 A2 1 A2 2 A2 3 A3 1 A3 2 A3 3 ;')
        model = AbstractModel()
        model.C = Set(dimen=2)
        instance = model.create_instance(currdir + 'loadC.dat')
        self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A1', 2), ('A1', 3), ('A2', 1), ('A2', 2), ('A2', 3), ('A3', 1), ('A3', 2), ('A3', 3)]))
        os.remove(currdir + 'loadC.dat')

    def test_tableC_2(self):
        with capture_output(currdir + 'loadC.dat'):
            print('table C={a,b} : a b := A1 1 A1 2 A1 3 A2 1 A2 2 A2 3 A3 1 A3 2 A3 3 ;')
        model = AbstractModel()
        model.C = Set(dimen=2)
        instance = model.create_instance(currdir + 'loadC.dat')
        self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A1', 2), ('A1', 3), ('A2', 1), ('A2', 2), ('A2', 3), ('A3', 1), ('A3', 2), ('A3', 3)]))
        os.remove(currdir + 'loadC.dat')

    def test_tableZ(self):
        with capture_output(currdir + 'loadZ.dat'):
            print('table Z := 1.01 ;')
        model = AbstractModel()
        model.Z = Param(default=99.0)
        instance = model.create_instance(currdir + 'loadZ.dat')
        self.assertEqual(instance.Z, 1.01)
        os.remove(currdir + 'loadZ.dat')

    def test_tableY_1(self):
        with capture_output(currdir + 'loadY.dat'):
            print('table columns=2 Y(1)={2} := A1 3.3 A2 3.4 A3 3.5 ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.Y = Param(model.A)
        instance = model.create_instance(currdir + 'loadY.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.Y.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        os.remove(currdir + 'loadY.dat')

    def test_tableY_2(self):
        with capture_output(currdir + 'loadY.dat'):
            print('table Y(A) : A Y := A1 3.3 A2 3.4 A3 3.5 ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.Y = Param(model.A)
        instance = model.create_instance(currdir + 'loadY.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.Y.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        os.remove(currdir + 'loadY.dat')

    def test_tableXW_1_1(self):
        with capture_output(currdir + 'loadXW.dat'):
            print('table columns=3 X(1)={2} W(1)={3} := A1 3.3 4.3 A2 3.4 4.4 A3 3.5 4.5 ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_1_2(self):
        with capture_output(currdir + 'loadXW.dat'):
            print('table X(A) W(A) : A X W := A1 3.3 4.3 A2 3.4 4.4 A3 3.5 4.5 ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_3_1(self):
        with capture_output(currdir + 'loadXW.dat'):
            print('table columns=3 A={1} X(A)={2} W(A)={3} := A1 3.3 4.3 A2 3.4 4.4 A3 3.5 4.5 ;')
        model = AbstractModel()
        model.A = Set()
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableXW_3_2(self):
        with capture_output(currdir + 'loadXW.dat'):
            print('table A={A} X(A) W(A) : A X W := A1 3.3 4.3 A2 3.4 4.4 A3 3.5 4.5 ;')
        model = AbstractModel()
        model.A = Set()
        model.X = Param(model.A)
        model.W = Param(model.A)
        instance = model.create_instance(currdir + 'loadXW.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
        os.remove(currdir + 'loadXW.dat')

    def test_tableS_1(self):
        with capture_output(currdir + 'loadS.dat'):
            print('table columns=2 S(1)={2} := A1 3.3 A2 . A3 3.5 ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.S = Param(model.A)
        instance = model.create_instance(currdir + 'loadS.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.S.extract_values(), {'A1': 3.3, 'A3': 3.5})
        os.remove(currdir + 'loadS.dat')

    def test_tableS_2(self):
        with capture_output(currdir + 'loadS.dat'):
            print('table S(A) : A S := A1 3.3 A2 . A3 3.5 ;')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.S = Param(model.A)
        instance = model.create_instance(currdir + 'loadS.dat')
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.S.extract_values(), {'A1': 3.3, 'A3': 3.5})
        os.remove(currdir + 'loadS.dat')

    def test_tablePO_1(self):
        with capture_output(currdir + 'loadPO.dat'):
            print('table columns=4 J={1,2} P(J)={3} O(J)={4} := A1 B1 4.3 5.3 A2 B2 4.4 5.4 A3 B3 4.5 5.5 ;')
        model = AbstractModel()
        model.J = Set(dimen=2)
        model.P = Param(model.J)
        model.O = Param(model.J)
        instance = model.create_instance(currdir + 'loadPO.dat')
        self.assertEqual(set(instance.J.data()), set([('A3', 'B3'), ('A1', 'B1'), ('A2', 'B2')]))
        self.assertEqual(instance.P.extract_values(), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})
        self.assertEqual(instance.O.extract_values(), {('A3', 'B3'): 5.5, ('A1', 'B1'): 5.3, ('A2', 'B2'): 5.4})
        os.remove(currdir + 'loadPO.dat')

    def test_tablePO_2(self):
        with capture_output(currdir + 'loadPO.dat'):
            print('table J={A,B} P(J) O(J) : A B P O := A1 B1 4.3 5.3 A2 B2 4.4 5.4 A3 B3 4.5 5.5 ;')
        model = AbstractModel()
        model.J = Set(dimen=2)
        model.P = Param(model.J)
        model.O = Param(model.J)
        instance = model.create_instance(currdir + 'loadPO.dat')
        self.assertEqual(set(instance.J.data()), set([('A3', 'B3'), ('A1', 'B1'), ('A2', 'B2')]))
        self.assertEqual(instance.P.extract_values(), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})
        self.assertEqual(instance.O.extract_values(), {('A3', 'B3'): 5.5, ('A1', 'B1'): 5.3, ('A2', 'B2'): 5.4})
        os.remove(currdir + 'loadPO.dat')

    def test_complex_1(self):
        with capture_output(currdir + 'loadComplex.dat'):
            print('table columns=8 I={4} J={3,5} A(I)={1} B(J)={7} :=')
            print('A1 x1 J311 I1 J321 y1 B1 z1')
            print('A2 x2 J312 I2 J322 y2 B2 z2')
            print('A3 x3 J313 I3 J323 y3 B3 z3')
            print(';')
        model = AbstractModel()
        model.I = Set()
        model.J = Set(dimen=2)
        model.A = Param(model.I)
        model.B = Param(model.J)
        instance = model.create_instance(currdir + 'loadComplex.dat')
        self.assertEqual(set(instance.J.data()), set([('J311', 'J321'), ('J312', 'J322'), ('J313', 'J323')]))
        self.assertEqual(set(instance.I.data()), set(['I1', 'I2', 'I3']))
        self.assertEqual(instance.B.extract_values(), {('J311', 'J321'): 'B1', ('J312', 'J322'): 'B2', ('J313', 'J323'): 'B3'})
        self.assertEqual(instance.A.extract_values(), {'I1': 'A1', 'I2': 'A2', 'I3': 'A3'})
        os.remove(currdir + 'loadComplex.dat')

    def test_complex_2(self):
        with capture_output(currdir + 'loadComplex.dat'):
            print('table I={I} J={J1,J2} A(J) B(I) :')
            print('A  x  J1   I  J2   y  B  z :=')
            print('A1 x1 J311 I1 J321 y1 B1 z1')
            print('A2 x2 J312 I2 J322 y2 B2 z2')
            print('A3 x3 J313 I3 J323 y3 B3 z3')
            print(';')
        model = AbstractModel()
        model.I = Set()
        model.J = Set(dimen=2)
        model.A = Param(model.J)
        model.B = Param(model.I)
        instance = model.create_instance(currdir + 'loadComplex.dat')
        self.assertEqual(set(instance.J.data()), set([('J311', 'J321'), ('J312', 'J322'), ('J313', 'J323')]))
        self.assertEqual(set(instance.I.data()), set(['I1', 'I2', 'I3']))
        self.assertEqual(instance.A.extract_values(), {('J311', 'J321'): 'A1', ('J312', 'J322'): 'A2', ('J313', 'J323'): 'A3'})
        self.assertEqual(instance.B.extract_values(), {'I1': 'B1', 'I2': 'B2', 'I3': 'B3'})
        os.remove(currdir + 'loadComplex.dat')