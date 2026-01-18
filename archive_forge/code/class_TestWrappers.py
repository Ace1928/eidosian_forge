import warnings
import base64
import ctypes
import pickle
import re
import subprocess
import sys
import weakref
import llvmlite.binding as ll
import unittest
from numba import njit
from numba.core.codegen import JITCPUCodegen
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase
class TestWrappers(TestCase):

    def test_noinline_on_main_call(self):

        @njit
        def foo():
            pass
        foo()
        sig = foo.signatures[0]
        ol = foo.overloads[sig]
        name = ol.fndesc.mangled_name.replace('$', '\\$')
        p1 = '.*call.*{}'.format(name)
        p2 = '.*(#[0-9]+).*'
        call_site = re.compile(p1 + p2)
        lines = foo.inspect_llvm(sig).splitlines()
        meta_data_idx = []
        for l in lines:
            matched = call_site.match(l)
            if matched:
                meta_data_idx.append(matched.groups()[0])
        self.assertEqual(len(meta_data_idx), 2)
        self.assertEqual(meta_data_idx[0], meta_data_idx[1])
        p1 = '^attributes\\s+{}'.format(meta_data_idx[0])
        p2 = '\\s+=\\s+{(.*)}.*$'
        attr_site = re.compile(p1 + p2)
        for l in reversed(lines):
            matched = attr_site.match(l)
            if matched:
                meta_data = matched.groups()[0]
                lmeta = meta_data.strip().split(' ')
                for x in lmeta:
                    if 'noinline' in x:
                        break
                else:
                    continue
                break
        else:
            return self.fail("Metadata did not match 'noinline'")