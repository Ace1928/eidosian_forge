import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
class TestGlobalValues(TestBase):

    def test_globals_access(self):
        mod = self.module()
        foo = ir.Function(mod, ir.FunctionType(ir.VoidType(), []), 'foo')
        ir.Function(mod, ir.FunctionType(ir.VoidType(), []), 'bar')
        globdouble = ir.GlobalVariable(mod, ir.DoubleType(), 'globdouble')
        self.assertEqual(mod.get_global('foo'), foo)
        self.assertEqual(mod.get_global('globdouble'), globdouble)
        with self.assertRaises(KeyError):
            mod.get_global('kkk')
        self.assertEqual(repr(globdouble), "<ir.GlobalVariable 'globdouble' of type 'double*'>")

    def test_functions_global_values_access(self):
        """
        Accessing functions and global values through Module.functions
        and Module.global_values.
        """
        mod = self.module()
        fty = ir.FunctionType(ir.VoidType(), [])
        foo = ir.Function(mod, fty, 'foo')
        bar = ir.Function(mod, fty, 'bar')
        globdouble = ir.GlobalVariable(mod, ir.DoubleType(), 'globdouble')
        self.assertEqual(set(mod.functions), set((foo, bar)))
        self.assertEqual(set(mod.global_values), set((foo, bar, globdouble)))

    def test_global_variables_ir(self):
        """
        IR serialization of global variables.
        """
        mod = self.module()
        a = ir.GlobalVariable(mod, int8, 'a')
        b = ir.GlobalVariable(mod, int8, 'b', addrspace=42)
        c = ir.GlobalVariable(mod, int32, 'c')
        c.initializer = int32(123)
        d = ir.GlobalVariable(mod, int32, 'd')
        d.global_constant = True
        e = ir.GlobalVariable(mod, int32, 'e')
        e.linkage = 'internal'
        f = ir.GlobalVariable(mod, int32, 'f', addrspace=456)
        f.unnamed_addr = True
        g = ir.GlobalVariable(mod, int32, 'g')
        g.linkage = 'internal'
        g.initializer = int32(123)
        g.align = 16
        h = ir.GlobalVariable(mod, int32, 'h')
        h.linkage = 'internal'
        h.initializer = int32(123)
        h.section = 'h_section'
        i = ir.GlobalVariable(mod, int32, 'i')
        i.linkage = 'internal'
        i.initializer = int32(456)
        i.align = 8
        i.section = 'i_section'
        self.check_module_body(mod, '            @"a" = external global i8\n            @"b" = external addrspace(42) global i8\n            @"c" = global i32 123\n            @"d" = external constant i32\n            @"e" = internal global i32 undef\n            @"f" = external unnamed_addr addrspace(456) global i32\n            @"g" = internal global i32 123, align 16\n            @"h" = internal global i32 123, section "h_section"\n            @"i" = internal global i32 456, section "i_section", align 8\n            ')

    def test_pickle(self):
        mod = self.module()
        self.assert_pickle_correctly(mod)