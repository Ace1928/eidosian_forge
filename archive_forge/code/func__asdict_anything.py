import copy
from ._compat import PY_3_9_PLUS, get_generic_base
from ._make import NOTHING, _obj_setattr, fields
from .exceptions import AttrsAttributeNotFoundError
def _asdict_anything(val, is_key, filter, dict_factory, retain_collection_types, value_serializer):
    """
    ``asdict`` only works on attrs instances, this works on anything.
    """
    if getattr(val.__class__, '__attrs_attrs__', None) is not None:
        rv = asdict(val, recurse=True, filter=filter, dict_factory=dict_factory, retain_collection_types=retain_collection_types, value_serializer=value_serializer)
    elif isinstance(val, (tuple, list, set, frozenset)):
        if retain_collection_types is True:
            cf = val.__class__
        elif is_key:
            cf = tuple
        else:
            cf = list
        rv = cf([_asdict_anything(i, is_key=False, filter=filter, dict_factory=dict_factory, retain_collection_types=retain_collection_types, value_serializer=value_serializer) for i in val])
    elif isinstance(val, dict):
        df = dict_factory
        rv = df(((_asdict_anything(kk, is_key=True, filter=filter, dict_factory=df, retain_collection_types=retain_collection_types, value_serializer=value_serializer), _asdict_anything(vv, is_key=False, filter=filter, dict_factory=df, retain_collection_types=retain_collection_types, value_serializer=value_serializer)) for kk, vv in val.items()))
    else:
        rv = val
        if value_serializer is not None:
            rv = value_serializer(None, None, rv)
    return rv