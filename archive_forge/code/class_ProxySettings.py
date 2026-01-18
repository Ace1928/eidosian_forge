import os
import pathlib
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Generic, TYPE_CHECKING
from lazyops.types.formatting import to_camel_case, to_snake_case, to_graphql_format
from lazyops.types.classprops import classproperty, lazyproperty
from lazyops.utils.serialization import Json
from pydantic import Field
from pydantic.networks import AnyUrl
from lazyops.imports._pydantic import BaseSettings as _BaseSettings
from lazyops.imports._pydantic import BaseModel as _BaseModel
from lazyops.imports._pydantic import (
class ProxySettings:

    def __init__(self, settings_cls: Optional[Type[BaseSettings]]=None, settings_getter: Optional[Union[Callable, str]]=None, debug_enabled: Optional[bool]=False):
        """
        Proxy settings object
        """
        assert settings_cls or settings_getter, 'Either settings_cls or settings_getter must be provided'
        self.__settings_cls_ = settings_cls
        self.__settings_getter_ = settings_getter
        if self.__settings_getter_ and isinstance(self.__settings_getter_, str):
            from lazyops.utils.helpers import import_string
            self.__settings_getter_ = import_string(self.__settings_getter_)
        self.__settings_ = None
        self.__debug_enabled_ = debug_enabled
        self.__last_attrs_: Dict[str, int] = {}

    @property
    def _settings_(self):
        """
        Returns the settings object
        """
        if self.__settings_ is None:
            if self.__settings_getter_:
                self.__settings_ = self.__settings_getter_()
            elif self.__settings_cls_:
                self.__settings_ = self.__settings_cls_()
        return self.__settings_

    def __getattr__(self, name):
        """
        Forward all unknown attributes to the settings object
        """
        if not self.__debug_enabled_:
            return getattr(self._settings_, name)
        if name not in self.__last_attrs_:
            self.__last_attrs_[name] = 0
        self.__last_attrs_[name] += 1
        if self.__last_attrs_[name] > 5:
            raise AttributeError(f'Settings object has no attribute {name}')
        if hasattr(self._settings_, name):
            self.__last_attrs_[name] = 0
            return getattr(self._settings_, name)
        raise AttributeError(f'Settings object has no attribute {name}')