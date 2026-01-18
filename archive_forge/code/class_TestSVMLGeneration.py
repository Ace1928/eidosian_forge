import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
@needs_svml
class TestSVMLGeneration(TestCase):
    """ Tests all SVML-generating functions produce desired calls """
    _numba_parallel_test_ = False
    asm_filter = re.compile('|'.join(['\\$[a-z_]\\w+,'] + list(svml_funcs)))

    @classmethod
    def mp_runner(cls, testname, outqueue):
        method = getattr(cls, testname)
        try:
            ok, msg = method()
        except Exception:
            msg = traceback.format_exc()
            ok = False
        outqueue.put({'status': ok, 'msg': msg})

    @classmethod
    def _inject_test(cls, dtype, mode, vlen, flags):
        if dtype.startswith('complex') and mode != 'numpy':
            return
        skipped = dtype.startswith('int') and vlen == 2
        sig = (numba.int64,)

        @staticmethod
        def run_template():
            fn, contains, avoids = combo_svml_usecase(dtype, mode, vlen, flags['fastmath'], flags['name'])
            with override_env_config('NUMBA_CPU_NAME', vlen2cpu[vlen]), override_env_config('NUMBA_CPU_FEATURES', vlen2cpu_features[vlen]):
                try:
                    jitted_fn = njit(sig, fastmath=flags['fastmath'], error_model=flags['error_model'])(fn)
                except:
                    raise Exception('raised while compiling ' + fn.__doc__)
            asm = jitted_fn.inspect_asm(sig)
            missed = [pattern for pattern in contains if not pattern in asm]
            found = [pattern for pattern in avoids if pattern in asm]
            ok = not missed and (not found)
            detail = '\n'.join([line for line in asm.split('\n') if cls.asm_filter.search(line) and (not '"' in line)])
            msg = f'While expecting {missed} and not {found},\nit contains:\n{detail}\nwhen compiling {fn.__doc__}'
            return (ok, msg)
        postfix = usecase_name(dtype, mode, vlen, flags['name'])
        testname = f'run_{postfix}'
        setattr(cls, testname, run_template)

        @unittest.skipUnless(not skipped, 'Not implemented')
        def test_runner(self):
            ctx = mp.get_context('spawn')
            q = ctx.Queue()
            p = ctx.Process(target=type(self).mp_runner, args=[testname, q])
            p.start()
            term_or_timeout = p.join(timeout=30)
            exitcode = p.exitcode
            if term_or_timeout is None:
                if exitcode is None:
                    self.fail('Process timed out.')
                elif exitcode < 0:
                    self.fail(f'Process terminated with signal {-exitcode}.')
            self.assertEqual(exitcode, 0, msg='process ended unexpectedly')
            out = q.get()
            status = out['status']
            msg = out['msg']
            self.assertTrue(status, msg=msg)
        setattr(cls, f'test_{postfix}', test_runner)

    @classmethod
    def autogenerate(cls):
        flag_list = [{'fastmath': False, 'error_model': 'numpy', 'name': 'usecase'}, {'fastmath': True, 'error_model': 'numpy', 'name': 'fastmath_usecase'}]
        for dtype in ('complex64', 'float64', 'float32', 'int32'):
            for vlen in vlen2cpu:
                for flags in flag_list:
                    for mode in ('scalar', 'range', 'prange', 'numpy'):
                        cls._inject_test(dtype, mode, vlen, dict(flags))
        for n in ('test_int32_range4_usecase',):
            setattr(cls, n, tag('important')(getattr(cls, n)))