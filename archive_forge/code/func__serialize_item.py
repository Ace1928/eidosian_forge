import numbers
import math
from ipywidgets import Widget, widget_serialization
def _serialize_item(value):
    if value is None:
        return {'value': None}
    elif isinstance(value, Widget):
        return {'type': 't', 'value': widget_serialization['to_json'](value, None)}
    elif isinstance(value, str):
        return {'type': 'c', 'value': value}
    elif isinstance(value, numbers.Real):
        if math.isnan(value) or math.isinf(value):
            value = repr(value)
        return {'value': value}
    elif isinstance(value, (list, tuple)):
        ret = {'value': list(value)}
        for i, e in enumerate(value):
            if isinstance(e, numbers.Real) and (math.isnan(e) or math.isinf(e)):
                ret['value'][i] = repr(e)
        ll = len(value)
        if 2 <= ll <= 4:
            ret['type'] = 'v%d' % ll
        elif ll == 9:
            ret['type'] = 'm3'
        elif ll == 16:
            ret['type'] = 'm4'
        return ret
    else:
        return {'value': value}