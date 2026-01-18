import math
import numpy as np
import sys
import ctypes
import warnings
from collections import namedtuple
import llvmlite.binding as ll
from llvmlite import ir
from numba import literal_unroll
from numba.core.extending import (
from numba.core import errors
from numba.core import types, utils
from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.randomimpl import (const_int, get_next_int, get_next_int32,
from ctypes import (  # noqa
def _gen_siphash(alg):
    if alg == 'siphash13':
        _ROUNDER = _SINGLE_ROUND
        _EXTRA_ROUND = True
    elif alg == 'siphash24':
        _ROUNDER = _DOUBLE_ROUND
        _EXTRA_ROUND = False
    else:
        assert 0, 'unreachable'

    @register_jitable(locals={'v0': types.uint64, 'v1': types.uint64, 'v2': types.uint64, 'v3': types.uint64, 'b': types.uint64, 'mi': types.uint64, 't': types.uint64, 'mask': types.uint64, 'jmp': types.uint64, 'ohexefef': types.uint64})
    def _siphash(k0, k1, src, src_sz):
        b = types.uint64(src_sz) << 56
        v0 = k0 ^ types.uint64(8317987319222330741)
        v1 = k1 ^ types.uint64(7237128888997146477)
        v2 = k0 ^ types.uint64(7816392313619706465)
        v3 = k1 ^ types.uint64(8387220255154660723)
        idx = 0
        while src_sz >= 8:
            mi = grab_uint64_t(src, idx)
            idx += 1
            src_sz -= 8
            v3 ^= mi
            v0, v1, v2, v3 = _ROUNDER(v0, v1, v2, v3)
            v0 ^= mi
        t = types.uint64(0)
        boffset = idx * 8
        ohexefef = types.uint64(255)
        if src_sz >= 7:
            jmp = 6 * 8
            mask = ~types.uint64(ohexefef << jmp)
            t = t & mask | types.uint64(grab_byte(src, boffset + 6)) << jmp
        if src_sz >= 6:
            jmp = 5 * 8
            mask = ~types.uint64(ohexefef << jmp)
            t = t & mask | types.uint64(grab_byte(src, boffset + 5)) << jmp
        if src_sz >= 5:
            jmp = 4 * 8
            mask = ~types.uint64(ohexefef << jmp)
            t = t & mask | types.uint64(grab_byte(src, boffset + 4)) << jmp
        if src_sz >= 4:
            t &= types.uint64(18446744069414584320)
            for i in range(4):
                jmp = i * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = t & mask | types.uint64(grab_byte(src, boffset + i)) << jmp
        if src_sz >= 3:
            jmp = 2 * 8
            mask = ~types.uint64(ohexefef << jmp)
            t = t & mask | types.uint64(grab_byte(src, boffset + 2)) << jmp
        if src_sz >= 2:
            jmp = 1 * 8
            mask = ~types.uint64(ohexefef << jmp)
            t = t & mask | types.uint64(grab_byte(src, boffset + 1)) << jmp
        if src_sz >= 1:
            mask = ~ohexefef
            t = t & mask | types.uint64(grab_byte(src, boffset + 0))
        b |= t
        v3 ^= b
        v0, v1, v2, v3 = _ROUNDER(v0, v1, v2, v3)
        v0 ^= b
        v2 ^= ohexefef
        v0, v1, v2, v3 = _ROUNDER(v0, v1, v2, v3)
        v0, v1, v2, v3 = _ROUNDER(v0, v1, v2, v3)
        if _EXTRA_ROUND:
            v0, v1, v2, v3 = _ROUNDER(v0, v1, v2, v3)
        t = v0 ^ v1 ^ (v2 ^ v3)
        return t
    return _siphash