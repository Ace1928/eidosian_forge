import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core
def _reduction_kernel_code(rk, filter_size, out_dtype, in_dtype):
    types = {}
    in_param, out_param = (rk.in_params[0], rk.out_params[0])
    in_ctype = _get_type_info(in_param, in_dtype, types)
    out_ctype = _get_type_info(out_param, out_dtype, types)
    types = '\n'.join(('typedef {} {};'.format(typ, name) for name, typ in types.items()))
    return 'namespace reduction_kernel {{\n{type_preamble}\n{preamble}\n__device__\nvoid {name}({in_const} CArray<{in_ctype}, 1, true, true>& _raw_{in_name},\n            CArray<{out_ctype}, 1, true, true>& _raw_{out_name}) {{\n    // these are just provided so if they are available for the RK\n    CIndexer<1> _in_ind({{{size}}});\n    CIndexer<0> _out_ind;\n\n    #define REDUCE(a, b) ({reduce_expr})\n    #define POST_MAP(a) ({post_map_expr})\n    typedef {reduce_type} _type_reduce;\n    _type_reduce _s = _type_reduce({identity});\n    for (int _j = 0; _j < {size}; ++_j) {{\n        _in_ind.set(_j);\n        {in_const} {in_ctype}& {in_name} = _raw_{in_name}[_j];\n        _type_reduce _a = static_cast<_type_reduce>({pre_map_expr});\n        _s = REDUCE(_s, _a);\n    }}\n    _out_ind.set(0);\n    {out_ctype} &{out_name} = _raw_{out_name}[0];\n    POST_MAP(_s);\n    #undef REDUCE\n    #undef POST_MAP\n}}\n}}'.format(name=rk.name, type_preamble=types, preamble=rk.preamble, in_const='const' if in_param.is_const else '', in_ctype=in_ctype, in_name=in_param.name, out_ctype=out_ctype, out_name=out_param.name, pre_map_expr=rk.map_expr, identity='' if rk.identity is None else rk.identity, size=filter_size, reduce_type=rk.reduce_type, reduce_expr=rk.reduce_expr, post_map_expr=rk.post_map_expr)