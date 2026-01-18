import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema
class _BaseDataclass:

    def _validate_field(self, key, val_type, required):
        value = getattr(self, key, None)
        if required and value is None:
            raise ValueError(f'`{key}` is required')
        if value is not None and (not isinstance(value, val_type)):
            raise ValueError(f'`{key}` must be of type {val_type.__name__}, got {type(value).__name__}')

    def _validate_list(self, key, val_type, required):
        values = getattr(self, key, None)
        if required and values is None:
            raise ValueError(f'`{key}` is required')
        if values is not None:
            if isinstance(values, list) and (not all((isinstance(v, val_type) for v in values))):
                raise ValueError(f'All items in `{key}` must be of type {val_type.__name__}')
            elif not isinstance(values, list):
                raise ValueError(f'`{key}` must be a list, got {type(values).__name__}')

    def _convert_dataclass_list(self, key, cls):
        values = getattr(self, key)
        if not isinstance(values, list):
            raise ValueError(f'`{key}` must be a list')
        if len(values) > 0:
            if all((isinstance(v, dict) for v in values)):
                try:
                    setattr(self, key, [cls(**v) for v in values])
                except TypeError as e:
                    raise ValueError(f'Error when coercing {values} to {cls.__name__}: {e}')
            elif any((not isinstance(v, cls) for v in values)):
                raise ValueError(f'Items in `{key}` must all have the same type: {cls.__name__} or dict')

    def to_dict(self):
        return asdict(self, dict_factory=lambda obj: {k: v for k, v in obj if v is not None})