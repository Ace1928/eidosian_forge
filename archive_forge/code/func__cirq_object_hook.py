import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _cirq_object_hook(d, resolvers: Sequence[JsonResolver], context_map: Dict[str, Any]):
    if 'cirq_type' not in d:
        return d
    if d['cirq_type'] == '_SerializedKey':
        return _SerializedKey.read_from_context(context_map, **d)
    if d['cirq_type'] == '_SerializedContext':
        _SerializedContext.update_context(context_map, **d)
        return None
    if d['cirq_type'] == '_ContextualSerialization':
        return _ContextualSerialization.deserialize_with_context(**d)
    cls = factory_from_json(d['cirq_type'], resolvers=resolvers)
    from_json_dict = getattr(cls, '_from_json_dict_', None)
    if from_json_dict is not None:
        return from_json_dict(**d)
    del d['cirq_type']
    return cls(**d)