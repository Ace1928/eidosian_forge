from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
class ClientProxyMetaClass(type):
    """
    A Custom Proxy that lazily defers initialization of a client until it is called
    """
    _clients: Dict[str, Union[ClientT, ClientTypeT]] = {}
    _module_name: str = None
    is_global: Optional[bool] = True

    @property
    def module_name(cls) -> str:
        """
        Returns the module name
        """
        if cls._module_name is None:
            cls._module_name = cls.__module__.split('.', 1)[0].strip()
        return cls._module_name

    def set_client_mapping(cls, mapping: Dict[str, str]):
        """
        Sets the client mapping
        """
        for key in mapping:
            if cls.module_name not in key:
                mapping[f'{cls.module_name}.{key}'] = mapping.pop(key)
        update_client_registry_mapping(mapping=mapping)

    def add_client_mapping(cls, name: str, module_path: str):
        """
        Adds a client mapping
        """
        if cls.module_name not in name:
            name = f'{cls.module_name}.{name}'
        update_client_registry_mapping(mapping={name: module_path})

    def get_or_init(cls, name: str, **kwargs) -> Union[ClientT, ClientTypeT]:
        """
        Initializes the client if it is not already initialized
        """
        if name not in cls._clients:
            key = f'{cls.module_name}.{name}' if cls.module_name not in name else name
            if cls.is_global:
                cls._clients[name] = get_global_client(name=key)
            else:
                cls._clients[name] = get_client(name=key, **kwargs)
        return cls._clients[name]

    def register(cls, client: Union[ClientT, ClientTypeT], name: str):
        """
        Registers a client
        """
        key = f'{cls.module_name}.{name}' if cls.module_name not in name else name
        if register_client(client=client, name=key) or name not in cls._clients:
            cls._clients[name] = client

    def __getitem__(cls, name: str) -> Union[ClientT, ClientTypeT]:
        """
        Returns a client
        """
        return cls.get_or_init(name=name)

    def __len__(cls) -> int:
        """
        Returns the number of clients
        """
        return len(cls._clients)