from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
def add_client_mapping(cls, name: str, module_path: str):
    """
        Adds a client mapping
        """
    if cls.module_name not in name:
        name = f'{cls.module_name}.{name}'
    update_client_registry_mapping(mapping={name: module_path})