import os
import pathlib
from typing import Any, Type, Tuple, Dict, List, Union, Optional, TYPE_CHECKING
from pydantic import Field, validator
from pydantic.networks import AnyUrl
from lazyops.types.formatting import to_camel_case, to_snake_case, to_graphql_format
from lazyops.types.classprops import classproperty, lazyproperty
from lazyops.utils.serialization import Json
@classproperty
def _model_fields(cls) -> List[str]:
    return [field.name for field in cls.__fields__.values()]