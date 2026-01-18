from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
def _parse_func_attr_value(key, value):
    """Converts a python object to an attr_value_pb2.AttrValue object."""
    if isinstance(value, attr_value_pb2.AttrValue):
        return value
    elif isinstance(value, bool):
        return attr_value_pb2.AttrValue(b=value)
    elif isinstance(value, int):
        return attr_value_pb2.AttrValue(i=value)
    elif isinstance(value, float):
        return attr_value_pb2.AttrValue(f=value)
    elif isinstance(value, (str, bytes)):
        return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
    elif isinstance(value, list):
        list_value = attr_value_pb2.AttrValue.ListValue()
        for v in value:
            if isinstance(v, bool):
                list_value.b.append(v)
            elif isinstance(v, int):
                list_value.i.append(v)
            elif isinstance(v, float):
                list_value.f.append(v)
            elif isinstance(v, (str, bytes)):
                list_value.s.append(compat.as_bytes(v))
            else:
                raise ValueError(f'Attributes for {key} must be bool, int, float, or string. Got {type(v)}.')
        return attr_value_pb2.AttrValue(list=list_value)
    else:
        raise ValueError(f'Attribute {key} must be bool, int, float, string, list, orAttrValue. Got {type(value)}.')