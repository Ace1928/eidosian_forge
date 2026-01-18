import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
def _register_known_types():
    f16 = ntypes.float16
    float16_ma = MachArLike(f16, machep=-10, negep=-11, minexp=-14, maxexp=16, it=10, iexp=5, ibeta=2, irnd=5, ngrd=0, eps=exp2(f16(-10)), epsneg=exp2(f16(-11)), huge=f16(65504), tiny=f16(2 ** (-14)))
    _register_type(float16_ma, b'f\xae')
    _float_ma[16] = float16_ma
    f32 = ntypes.float32
    float32_ma = MachArLike(f32, machep=-23, negep=-24, minexp=-126, maxexp=128, it=23, iexp=8, ibeta=2, irnd=5, ngrd=0, eps=exp2(f32(-23)), epsneg=exp2(f32(-24)), huge=f32((1 - 2 ** (-24)) * 2 ** 128), tiny=exp2(f32(-126)))
    _register_type(float32_ma, b'\xcd\xcc\xcc\xbd')
    _float_ma[32] = float32_ma
    f64 = ntypes.float64
    epsneg_f64 = 2.0 ** (-53.0)
    tiny_f64 = 2.0 ** (-1022.0)
    float64_ma = MachArLike(f64, machep=-52, negep=-53, minexp=-1022, maxexp=1024, it=52, iexp=11, ibeta=2, irnd=5, ngrd=0, eps=2.0 ** (-52.0), epsneg=epsneg_f64, huge=(1.0 - epsneg_f64) / tiny_f64 * f64(4), tiny=tiny_f64)
    _register_type(float64_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _float_ma[64] = float64_ma
    ld = ntypes.longdouble
    epsneg_f128 = exp2(ld(-113))
    tiny_f128 = exp2(ld(-16382))
    with numeric.errstate(all='ignore'):
        huge_f128 = (ld(1) - epsneg_f128) / tiny_f128 * ld(4)
    float128_ma = MachArLike(ld, machep=-112, negep=-113, minexp=-16382, maxexp=16384, it=112, iexp=15, ibeta=2, irnd=5, ngrd=0, eps=exp2(ld(-112)), epsneg=epsneg_f128, huge=huge_f128, tiny=tiny_f128)
    _register_type(float128_ma, b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    _float_ma[128] = float128_ma
    epsneg_f80 = exp2(ld(-64))
    tiny_f80 = exp2(ld(-16382))
    with numeric.errstate(all='ignore'):
        huge_f80 = (ld(1) - epsneg_f80) / tiny_f80 * ld(4)
    float80_ma = MachArLike(ld, machep=-63, negep=-64, minexp=-16382, maxexp=16384, it=63, iexp=15, ibeta=2, irnd=5, ngrd=0, eps=exp2(ld(-63)), epsneg=epsneg_f80, huge=huge_f80, tiny=tiny_f80)
    _register_type(float80_ma, b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf')
    _float_ma[80] = float80_ma
    huge_dd = nextafter(ld(inf), ld(0), dtype=ld)
    smallest_normal_dd = NaN
    smallest_subnormal_dd = ld(nextafter(0.0, 1.0))
    float_dd_ma = MachArLike(ld, machep=-105, negep=-106, minexp=-1022, maxexp=1024, it=105, iexp=11, ibeta=2, irnd=5, ngrd=0, eps=exp2(ld(-105)), epsneg=exp2(ld(-106)), huge=huge_dd, tiny=smallest_normal_dd, smallest_subnormal=smallest_subnormal_dd)
    _register_type(float_dd_ma, b'\x9a\x99\x99\x99\x99\x99Y<\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _register_type(float_dd_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf\x9a\x99\x99\x99\x99\x99Y<')
    _float_ma['dd'] = float_dd_ma