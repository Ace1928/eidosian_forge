import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
class TestConstant(TestBase):

    def test_integers(self):
        c = ir.Constant(int32, 42)
        self.assertEqual(str(c), 'i32 42')
        c = ir.Constant(int1, 1)
        self.assertEqual(str(c), 'i1 1')
        c = ir.Constant(int1, 0)
        self.assertEqual(str(c), 'i1 0')
        c = ir.Constant(int1, True)
        self.assertEqual(str(c), 'i1 true')
        c = ir.Constant(int1, False)
        self.assertEqual(str(c), 'i1 false')
        c = ir.Constant(int1, ir.Undefined)
        self.assertEqual(str(c), 'i1 undef')
        c = ir.Constant(int1, None)
        self.assertEqual(str(c), 'i1 0')

    def test_reals(self):
        c = ir.Constant(flt, 1.5)
        self.assertEqual(str(c), 'float 0x3ff8000000000000')
        c = ir.Constant(flt, -1.5)
        self.assertEqual(str(c), 'float 0xbff8000000000000')
        c = ir.Constant(dbl, 1.5)
        self.assertEqual(str(c), 'double 0x3ff8000000000000')
        c = ir.Constant(dbl, -1.5)
        self.assertEqual(str(c), 'double 0xbff8000000000000')
        c = ir.Constant(dbl, ir.Undefined)
        self.assertEqual(str(c), 'double undef')
        c = ir.Constant(dbl, None)
        self.assertEqual(str(c), 'double 0.0')

    def test_arrays(self):
        c = ir.Constant(ir.ArrayType(int32, 3), (c32(5), c32(6), c32(4)))
        self.assertEqual(str(c), '[3 x i32] [i32 5, i32 6, i32 4]')
        c = ir.Constant(ir.ArrayType(int32, 2), (c32(5), c32(ir.Undefined)))
        self.assertEqual(str(c), '[2 x i32] [i32 5, i32 undef]')
        c = ir.Constant.literal_array((c32(5), c32(6), c32(ir.Undefined)))
        self.assertEqual(str(c), '[3 x i32] [i32 5, i32 6, i32 undef]')
        with self.assertRaises(TypeError) as raises:
            ir.Constant.literal_array((c32(5), ir.Constant(flt, 1.5)))
        self.assertEqual(str(raises.exception), 'all elements must have the same type')
        c = ir.Constant(ir.ArrayType(int32, 2), ir.Undefined)
        self.assertEqual(str(c), '[2 x i32] undef')
        c = ir.Constant(ir.ArrayType(int32, 2), None)
        self.assertEqual(str(c), '[2 x i32] zeroinitializer')
        c = ir.Constant(ir.ArrayType(int8, 11), bytearray(b'foobar_123\x80'))
        self.assertEqual(str(c), '[11 x i8] c"foobar_123\\80"')
        c = ir.Constant(ir.ArrayType(int8, 4), bytearray(b'\x00\x01\x04\xff'))
        self.assertEqual(str(c), '[4 x i8] c"\\00\\01\\04\\ff"')
        c = ir.Constant(ir.ArrayType(int32, 3), (5, ir.Undefined, 6))
        self.assertEqual(str(c), '[3 x i32] [i32 5, i32 undef, i32 6]')
        with self.assertRaises(ValueError):
            ir.Constant(ir.ArrayType(int32, 3), (5, 6))

    def test_vector(self):
        vecty = ir.VectorType(ir.IntType(32), 8)
        vals = [1, 2, 4, 3, 8, 6, 9, 7]
        vec = ir.Constant(vecty, vals)
        vec_repr = '<8 x i32> <{}>'.format(', '.join(map('i32 {}'.format, vals)))
        self.assertEqual(str(vec), vec_repr)

    def test_non_nullable_int(self):
        constant = ir.Constant(ir.IntType(32), None).constant
        self.assertEqual(constant, 0)

    def test_structs(self):
        st1 = ir.LiteralStructType((flt, int1))
        st2 = ir.LiteralStructType((int32, st1))
        c = ir.Constant(st1, (ir.Constant(ir.FloatType(), 1.5), ir.Constant(int1, True)))
        self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 true}')
        c = ir.Constant.literal_struct((ir.Constant(ir.FloatType(), 1.5), ir.Constant(int1, True)))
        self.assertEqual(c.type, st1)
        self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 true}')
        c = ir.Constant.literal_struct((ir.Constant(ir.FloatType(), 1.5), ir.Constant(int1, ir.Undefined)))
        self.assertEqual(c.type, st1)
        self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 undef}')
        c = ir.Constant(st1, ir.Undefined)
        self.assertEqual(str(c), '{float, i1} undef')
        c = ir.Constant(st1, None)
        self.assertEqual(str(c), '{float, i1} zeroinitializer')
        c1 = ir.Constant(st1, (1.5, True))
        self.assertEqual(str(c1), '{float, i1} {float 0x3ff8000000000000, i1 true}')
        c2 = ir.Constant(st2, (42, c1))
        self.assertEqual(str(c2), '{i32, {float, i1}} {i32 42, {float, i1} {float 0x3ff8000000000000, i1 true}}')
        c3 = ir.Constant(st2, (42, (1.5, True)))
        self.assertEqual(str(c3), str(c2))
        with self.assertRaises(ValueError):
            ir.Constant(st2, (4, 5, 6))

    def test_undefined_literal_struct_pickling(self):
        i8 = ir.IntType(8)
        st = ir.Constant(ir.LiteralStructType([i8, i8]), ir.Undefined)
        self.assert_pickle_correctly(st)

    def test_type_instantiaton(self):
        """
        Instantiating a type should create a constant.
        """
        c = int8(42)
        self.assertIsInstance(c, ir.Constant)
        self.assertEqual(str(c), 'i8 42')
        c = int1(True)
        self.assertIsInstance(c, ir.Constant)
        self.assertEqual(str(c), 'i1 true')
        at = ir.ArrayType(int32, 3)
        c = at([c32(4), c32(5), c32(6)])
        self.assertEqual(str(c), '[3 x i32] [i32 4, i32 5, i32 6]')
        c = at([4, 5, 6])
        self.assertEqual(str(c), '[3 x i32] [i32 4, i32 5, i32 6]')
        c = at(None)
        self.assertEqual(str(c), '[3 x i32] zeroinitializer')
        with self.assertRaises(ValueError):
            at([4, 5, 6, 7])
        st1 = ir.LiteralStructType((flt, int1))
        st2 = ir.LiteralStructType((int32, st1))
        c = st1((1.5, True))
        self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 true}')
        c = st2((42, (1.5, True)))
        self.assertEqual(str(c), '{i32, {float, i1}} {i32 42, {float, i1} {float 0x3ff8000000000000, i1 true}}')

    def test_repr(self):
        """
        Constants should have a useful repr().
        """
        c = int32(42)
        self.assertEqual(repr(c), "<ir.Constant type='i32' value=42>")

    def test_encoding_problem(self):
        c = ir.Constant(ir.ArrayType(ir.IntType(8), 256), bytearray(range(256)))
        m = self.module()
        gv = ir.GlobalVariable(m, c.type, 'myconstant')
        gv.global_constant = True
        gv.initializer = c
        parsed = llvm.parse_assembly(str(m))
        reparsed = llvm.parse_assembly(str(parsed))
        self.assertEqual(str(parsed), str(reparsed))

    def test_gep(self):
        m = self.module()
        tp = ir.LiteralStructType((flt, int1))
        gv = ir.GlobalVariable(m, tp, 'myconstant')
        c = gv.gep([ir.Constant(int32, x) for x in (0, 1)])
        self.assertEqual(str(c), 'getelementptr ({float, i1}, {float, i1}* @"myconstant", i32 0, i32 1)')
        self.assertEqual(c.type, ir.PointerType(int1))
        const = ir.Constant(tp, None)
        with self.assertRaises(TypeError):
            const.gep([ir.Constant(int32, 0)])
        const_ptr = ir.Constant(tp.as_pointer(), None)
        c2 = const_ptr.gep([ir.Constant(int32, 0)])
        self.assertEqual(str(c2), 'getelementptr ({float, i1}, {float, i1}* null, i32 0)')
        self.assertEqual(c.type, ir.PointerType(int1))

    def test_gep_addrspace_globalvar(self):
        m = self.module()
        tp = ir.LiteralStructType((flt, int1))
        addrspace = 4
        gv = ir.GlobalVariable(m, tp, 'myconstant', addrspace=addrspace)
        self.assertEqual(gv.addrspace, addrspace)
        c = gv.gep([ir.Constant(int32, x) for x in (0, 1)])
        self.assertEqual(c.type.addrspace, addrspace)
        self.assertEqual(str(c), 'getelementptr ({float, i1}, {float, i1} addrspace(4)* @"myconstant", i32 0, i32 1)')
        self.assertEqual(c.type, ir.PointerType(int1, addrspace=addrspace))

    def test_trunc(self):
        c = ir.Constant(int64, 1).trunc(int32)
        self.assertEqual(str(c), 'trunc (i64 1 to i32)')

    def test_zext(self):
        c = ir.Constant(int32, 1).zext(int64)
        self.assertEqual(str(c), 'zext (i32 1 to i64)')

    def test_sext(self):
        c = ir.Constant(int32, -1).sext(int64)
        self.assertEqual(str(c), 'sext (i32 -1 to i64)')

    def test_fptrunc(self):
        c = ir.Constant(flt, 1).fptrunc(hlf)
        self.assertEqual(str(c), 'fptrunc (float 0x3ff0000000000000 to half)')

    def test_fpext(self):
        c = ir.Constant(flt, 1).fpext(dbl)
        self.assertEqual(str(c), 'fpext (float 0x3ff0000000000000 to double)')

    def test_bitcast(self):
        m = self.module()
        gv = ir.GlobalVariable(m, int32, 'myconstant')
        c = gv.bitcast(int64.as_pointer())
        self.assertEqual(str(c), 'bitcast (i32* @"myconstant" to i64*)')

    def test_fptoui(self):
        c = ir.Constant(flt, 1).fptoui(int32)
        self.assertEqual(str(c), 'fptoui (float 0x3ff0000000000000 to i32)')

    def test_uitofp(self):
        c = ir.Constant(int32, 1).uitofp(flt)
        self.assertEqual(str(c), 'uitofp (i32 1 to float)')

    def test_fptosi(self):
        c = ir.Constant(flt, 1).fptosi(int32)
        self.assertEqual(str(c), 'fptosi (float 0x3ff0000000000000 to i32)')

    def test_sitofp(self):
        c = ir.Constant(int32, 1).sitofp(flt)
        self.assertEqual(str(c), 'sitofp (i32 1 to float)')

    def test_ptrtoint_1(self):
        ptr = ir.Constant(int64.as_pointer(), None)
        one = ir.Constant(int32, 1)
        c = ptr.ptrtoint(int32)
        self.assertRaises(TypeError, one.ptrtoint, int64)
        self.assertRaises(TypeError, ptr.ptrtoint, flt)
        self.assertEqual(str(c), 'ptrtoint (i64* null to i32)')

    def test_ptrtoint_2(self):
        m = self.module()
        gv = ir.GlobalVariable(m, int32, 'myconstant')
        c = gv.ptrtoint(int64)
        self.assertEqual(str(c), 'ptrtoint (i32* @"myconstant" to i64)')
        self.assertRaisesRegex(TypeError, "can only ptrtoint\\(\\) to integer type, not 'i64\\*'", gv.ptrtoint, int64.as_pointer())
        c2 = ir.Constant(int32, 0)
        self.assertRaisesRegex(TypeError, "can only call ptrtoint\\(\\) on pointer type, not 'i32'", c2.ptrtoint, int64)

    def test_inttoptr(self):
        one = ir.Constant(int32, 1)
        pi = ir.Constant(flt, 3.14)
        c = one.inttoptr(int64.as_pointer())
        self.assertRaises(TypeError, one.inttoptr, int64)
        self.assertRaises(TypeError, pi.inttoptr, int64.as_pointer())
        self.assertEqual(str(c), 'inttoptr (i32 1 to i64*)')

    def test_neg(self):
        one = ir.Constant(int32, 1)
        self.assertEqual(str(one.neg()), 'sub (i32 0, i32 1)')

    def test_not(self):
        one = ir.Constant(int32, 1)
        self.assertEqual(str(one.not_()), 'xor (i32 1, i32 -1)')

    def test_fneg(self):
        one = ir.Constant(flt, 1)
        self.assertEqual(str(one.fneg()), 'fneg (float 0x3ff0000000000000)')

    def test_int_binops(self):
        one = ir.Constant(int32, 1)
        two = ir.Constant(int32, 2)
        oracle = {one.shl: 'shl', one.lshr: 'lshr', one.ashr: 'ashr', one.add: 'add', one.sub: 'sub', one.mul: 'mul', one.udiv: 'udiv', one.sdiv: 'sdiv', one.urem: 'urem', one.srem: 'srem', one.or_: 'or', one.and_: 'and', one.xor: 'xor'}
        for fn, irop in oracle.items():
            self.assertEqual(str(fn(two)), irop + ' (i32 1, i32 2)')
        oracle = {'==': 'eq', '!=': 'ne', '>': 'ugt', '>=': 'uge', '<': 'ult', '<=': 'ule'}
        for cop, cond in oracle.items():
            actual = str(one.icmp_unsigned(cop, two))
            expected = 'icmp ' + cond + ' (i32 1, i32 2)'
            self.assertEqual(actual, expected)
        oracle = {'==': 'eq', '!=': 'ne', '>': 'sgt', '>=': 'sge', '<': 'slt', '<=': 'sle'}
        for cop, cond in oracle.items():
            actual = str(one.icmp_signed(cop, two))
            expected = 'icmp ' + cond + ' (i32 1, i32 2)'
            self.assertEqual(actual, expected)

    def test_flt_binops(self):
        one = ir.Constant(flt, 1)
        two = ir.Constant(flt, 2)
        oracle = {one.fadd: 'fadd', one.fsub: 'fsub', one.fmul: 'fmul', one.fdiv: 'fdiv', one.frem: 'frem'}
        for fn, irop in oracle.items():
            actual = str(fn(two))
            expected = irop + ' (float 0x3ff0000000000000, float 0x4000000000000000)'
            self.assertEqual(actual, expected)
        oracle = {'==': 'oeq', '!=': 'one', '>': 'ogt', '>=': 'oge', '<': 'olt', '<=': 'ole'}
        for cop, cond in oracle.items():
            actual = str(one.fcmp_ordered(cop, two))
            expected = 'fcmp ' + cond + ' (float 0x3ff0000000000000, float 0x4000000000000000)'
            self.assertEqual(actual, expected)
        oracle = {'==': 'ueq', '!=': 'une', '>': 'ugt', '>=': 'uge', '<': 'ult', '<=': 'ule'}
        for cop, cond in oracle.items():
            actual = str(one.fcmp_unordered(cop, two))
            expected = 'fcmp ' + cond + ' (float 0x3ff0000000000000, float 0x4000000000000000)'
            self.assertEqual(actual, expected)