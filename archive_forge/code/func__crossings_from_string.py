import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _crossings_from_string(self, spec):
    """
        >>> Link('T(3, 2)')
        <Link: 1 comp; 4 cross>
        >>> Link('DT: [(4,6,2)]')
        <Link: 1 comp; 3 cross>
        >>> Link('DT: [(4,6,2)], [0,0,1]')
        <Link: 1 comp; 3 cross>
        >>> Link('DT: cacbca.001')
        <Link: 1 comp; 3 cross>
        >>> Link('DT: cacbca')
        <Link: 1 comp; 3 cross>
        >>> Link('DT[cacbca.001]')
        <Link: 1 comp; 3 cross>
        """
    if spec.startswith('T('):
        from . import torus
        crossings = torus.torus_knot(spec).crossings
    else:
        from ..codecs import DTcodec
        m = is_int_DT_exterior.match(spec)
        if m:
            code = eval(m.group(1), {})
            if isinstance(code, tuple):
                dt = DTcodec(*code)
            elif isinstance(code, list) and isinstance(code[0], int):
                dt = DTcodec([tuple(code)])
            else:
                dt = DTcodec(code)
        else:
            m = is_alpha_DT_exterior.match(spec)
            if m:
                dt = DTcodec(m.group(1))
            else:
                dt_string = lookup_DT_code_by_name(spec)
                if dt_string is None:
                    raise ValueError('No link by that name known')
                self.name = spec
                dt = DTcodec(dt_string)
        crossings = dt.PD_code()
    return crossings