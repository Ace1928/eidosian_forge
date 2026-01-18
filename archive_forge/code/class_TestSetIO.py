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
class TestSetIO(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)

    def tearDown(self):
        if os.path.exists(currdir + 'setA.dat'):
            os.remove(currdir + 'setA.dat')
        PyomoModel.tearDown(self)

    def test_io1(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set A := A1 A2 A3; end;')
        OUTPUT.close()
        self.model.A = Set()
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.A), 3)

    def test_io2(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set B := 1 2 3 4; end;')
        OUTPUT.close()
        self.model.B = Set()
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.B), 4)

    def test_io3(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := A1 A2 A3;\n')
        OUTPUT.write('set B := 1 2 3 4;\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.B = Set()
        self.model.C = self.model.A * self.model.B
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.C), 12)

    def test_io3a(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := A1 A2 A3;\n')
        OUTPUT.write('set B := 1 2 3 4;\n')
        OUTPUT.write('set C := (A1,1) (A2,2) (A3,3);\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.B = Set()
        self.model.C = self.model.A * self.model.B
        with self.assertRaisesRegex(ValueError, 'SetOperator C with incompatible data'):
            self.instance = self.model.create_instance(currdir + 'setA.dat')

    def test_io4(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := A1 A2 A3;\n')
        OUTPUT.write('set B := 1 2 3 4;\n')
        OUTPUT.write('set D := (A1,1) (A2,2) (A3,3);\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.B = Set()
        self.model.D = Set(within=self.model.A * self.model.B)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.D), 3)

    def test_io5(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := A1 A2 A3;\n')
        OUTPUT.write('set B := 1 2 3 4;\n')
        OUTPUT.write('set D : A1 A2 A3 :=\n')
        OUTPUT.write('    1   +  -  +\n')
        OUTPUT.write('    2   +  -  +\n')
        OUTPUT.write('    3   +  -  +\n')
        OUTPUT.write('    4   +  -  +;\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.B = Set()
        self.model.D = Set(within=self.model.A * self.model.B)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.D), 8)

    def test_io6(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := A1 A2 A3;\n')
        OUTPUT.write('set B := 1 2 3 4;\n')
        OUTPUT.write('set E :=\n')
        OUTPUT.write('(A1,1,*) A1 A2\n')
        OUTPUT.write('(A2,2,*) A2 A3\n')
        OUTPUT.write('(A3,3,*) A1 A3 ;\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.B = Set()
        self.model.E = Set(within=self.model.A * self.model.B * self.model.A)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.E), 6)

    def test_io7(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := A1 A2 A3;\n')
        OUTPUT.write('set B := 1 2 3 4;\n')
        OUTPUT.write('set F[A1] := 1 3 5;\n')
        OUTPUT.write('set F[A2] := 2 4 6;\n')
        OUTPUT.write('set F[A3] := 3 5 7;\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.B = Set()
        self.model.F = Set(self.model.A)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(self.instance.F.dim(), 1)
        self.assertEqual(len(list(self.instance.F.keys())), 3)
        self.assertEqual(len(self.instance.F['A1']), 3)

    def test_io8(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set E :=\n')
        OUTPUT.write('(A1,1,*) A1 A2\n')
        OUTPUT.write('(*,2,*) A2 A3\n')
        OUTPUT.write('(A3,3,*) A1 A3 ;\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.E = Set(dimen=3)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.E), 5)

    def test_io9(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set E :=\n')
        OUTPUT.write('(A1,1,A1) (A1,1,A2)\n')
        OUTPUT.write('(A2,2,A3)\n')
        OUTPUT.write('(A3,3,A1) (A3,3,A3) ;\n')
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.E = Set(dimen=3)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.E), 5)

    def test_io10(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data;\n')
        OUTPUT.write('set A := \'A1 x\' \' A2\' "A3";\n')
        OUTPUT.write("set F['A1 x'] := 1 3 5;\n")
        OUTPUT.write('set F[" A2"] := 2 4 6;\n')
        OUTPUT.write("set F['A3'] := 3 5 7;\n")
        OUTPUT.write('end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.F = Set(self.model.A)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(self.instance.F.dim(), 1)
        self.assertEqual(len(list(self.instance.F.keys())), 3)
        self.assertEqual(len(self.instance.F['A1 x']), 3)