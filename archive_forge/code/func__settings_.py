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