import re
from .utilities import MethodMappingList
from .component import NonZeroDimensionalComponent
from .coordinates import PtolemyCoordinates
from .rur import RUR
from . import processFileBase
from ..pari import pari
def _process_rur_component(text, py_eval, manifold_thunk):
    lines = text.split('\n')
    dimension = None
    is_prime = False
    format = None
    body = ''
    for line in lines:
        m = re.match('==(.*):(.*)', line)
        if m:
            key, val = m.groups()
            val = val.strip()
            if key == 'DIMENSION':
                dimension = int(val)
            elif key == 'IS_PRIME':
                if val == 'TRUE':
                    is_prime = True
                elif val != 'FALSE':
                    raise Exception('IS_PRIME needs to be TRUE or FALSE')
            elif key == 'FORMAT':
                format = val
            else:
                raise Exception('Unrecognized key %s' % key)
        else:
            body += line + '\n'
    if dimension is None:
        return NonZeroDimensionalComponent()
    if dimension > 0:
        return NonZeroDimensionalComponent(dimension=dimension)
    if format is None:
        raise Exception('No format specified')
    if format == 'MAPLE-LIKE':
        d = parse_maple_like_rur(body.strip())
        return PtolemyCoordinates(d, is_numerical=False, py_eval_section=py_eval, manifold_thunk=manifold_thunk)
    else:
        raise Exception('Unknown format %s' % format)