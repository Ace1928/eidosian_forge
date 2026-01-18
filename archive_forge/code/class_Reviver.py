import importlib
import json
import os
from typing import Any, Dict, List, Optional
from langchain_core._api import beta
from langchain_core.load.mapping import (
from langchain_core.load.serializable import Serializable
class Reviver:
    """Reviver for JSON objects."""

    def __init__(self, secrets_map: Optional[Dict[str, str]]=None, valid_namespaces: Optional[List[str]]=None, secrets_from_env: bool=True) -> None:
        self.secrets_from_env = secrets_from_env
        self.secrets_map = secrets_map or dict()
        self.valid_namespaces = [*DEFAULT_NAMESPACES, *valid_namespaces] if valid_namespaces else DEFAULT_NAMESPACES

    def __call__(self, value: Dict[str, Any]) -> Any:
        if value.get('lc', None) == 1 and value.get('type', None) == 'secret' and (value.get('id', None) is not None):
            [key] = value['id']
            if key in self.secrets_map:
                return self.secrets_map[key]
            else:
                if self.secrets_from_env and key in os.environ and os.environ[key]:
                    return os.environ[key]
                raise KeyError(f'Missing key "{key}" in load(secrets_map)')
        if value.get('lc', None) == 1 and value.get('type', None) == 'not_implemented' and (value.get('id', None) is not None):
            raise NotImplementedError(f"Trying to load an object that doesn't implement serialization: {value}")
        if value.get('lc', None) == 1 and value.get('type', None) == 'constructor' and (value.get('id', None) is not None):
            [*namespace, name] = value['id']
            if namespace[0] not in self.valid_namespaces:
                raise ValueError(f'Invalid namespace: {value}')
            if len(namespace) == 1 and namespace[0] == 'langchain':
                raise ValueError(f'Invalid namespace: {value}')
            if namespace[0] in DEFAULT_NAMESPACES:
                key = tuple(namespace + [name])
                if key not in ALL_SERIALIZABLE_MAPPINGS:
                    raise ValueError(f'Trying to deserialize something that cannot be deserialized in current version of langchain-core: {key}')
                import_path = ALL_SERIALIZABLE_MAPPINGS[key]
                import_dir, import_obj = (import_path[:-1], import_path[-1])
                mod = importlib.import_module('.'.join(import_dir))
                cls = getattr(mod, import_obj)
            else:
                mod = importlib.import_module('.'.join(namespace))
                cls = getattr(mod, name)
            if not issubclass(cls, Serializable):
                raise ValueError(f'Invalid namespace: {value}')
            kwargs = value.get('kwargs', dict())
            return cls(**kwargs)
        return value