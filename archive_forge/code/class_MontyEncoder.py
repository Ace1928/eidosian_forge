from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
class MontyEncoder(json.JSONEncoder):
    """
    A Json Encoder which supports the MSONable API, plus adds support for
    numpy arrays, datetime objects, bson ObjectIds (requires bson).
    Usage::
        # Add it as a *cls* keyword when using json.dump
        json.dumps(object, cls=MontyEncoder)
    """

    def default(self, o) -> dict:
        """
        Overriding default method for JSON encoding. This method does two
        things: (a) If an object has a to_dict property, return the to_dict
        output. (b) If the @module and @class keys are not in the to_dict,
        add them to the output automatically. If the object has no to_dict
        property, the default Python json encoder default method is called.
        Args:
            o: Python object.
        Return:
            Python dict representation.
        """
        if isinstance(o, datetime.datetime):
            return {'@module': 'datetime', '@class': 'datetime', 'string': str(o)}
        if isinstance(o, UUID):
            return {'@module': 'uuid', '@class': 'UUID', 'string': str(o)}
        if isinstance(o, Path):
            return {'@module': 'pathlib', '@class': 'Path', 'string': str(o)}
        if torch is not None and isinstance(o, torch.Tensor):
            d = {'@module': 'torch', '@class': 'Tensor', 'dtype': o.type()}
            if 'Complex' in o.type():
                d['data'] = [o.real.tolist(), o.imag.tolist()]
            else:
                d['data'] = o.numpy().tolist()
            return d
        if np is not None:
            if isinstance(o, np.ndarray):
                if str(o.dtype).startswith('complex'):
                    return {'@module': 'numpy', '@class': 'array', 'dtype': str(o.dtype), 'data': [o.real.tolist(), o.imag.tolist()]}
                return {'@module': 'numpy', '@class': 'array', 'dtype': str(o.dtype), 'data': o.tolist()}
            if isinstance(o, np.generic):
                return o.item()
        if _check_type(o, 'pandas.core.frame.DataFrame'):
            return {'@module': 'pandas', '@class': 'DataFrame', 'data': o.to_json(default_handler=MontyEncoder().encode)}
        if _check_type(o, 'pandas.core.series.Series'):
            return {'@module': 'pandas', '@class': 'Series', 'data': o.to_json(default_handler=MontyEncoder().encode)}
        if bson is not None and isinstance(o, bson.objectid.ObjectId):
            return {'@module': 'bson.objectid', '@class': 'ObjectId', 'oid': str(o)}
        if callable(o) and (not isinstance(o, MSONable)):
            return _serialize_callable(o)
        try:
            if pydantic is not None and isinstance(o, pydantic.BaseModel):
                d = o.dict()
            elif dataclasses is not None and (not issubclass(o.__class__, MSONable)) and dataclasses.is_dataclass(o):
                d = dataclasses.asdict(o)
            elif hasattr(o, 'as_dict'):
                d = o.as_dict()
            elif isinstance(o, Enum):
                d = {'value': o.value}
            else:
                raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
            if '@module' not in d:
                d['@module'] = str(o.__class__.__module__)
            if '@class' not in d:
                d['@class'] = str(o.__class__.__name__)
            if '@version' not in d:
                try:
                    parent_module = o.__class__.__module__.split('.')[0]
                    module_version = import_module(parent_module).__version__
                    d['@version'] = str(module_version)
                except (AttributeError, ImportError):
                    d['@version'] = None
            return d
        except AttributeError:
            return json.JSONEncoder.default(self, o)