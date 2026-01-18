from __future__ import annotations as _annotations
import warnings
from contextlib import contextmanager
from typing import (
from pydantic_core import core_schema
from typing_extensions import (
from ..aliases import AliasGenerator
from ..config import ConfigDict, ExtraValues, JsonDict, JsonEncoder, JsonSchemaExtraCallable
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
Create a pydantic-core config, `obj` is just used to populate `title` if not set in config.

        Pass `obj=None` if you do not want to attempt to infer the `title`.

        We don't use getattr here since we don't want to populate with defaults.

        Args:
            obj: An object used to populate `title` if not set in config.

        Returns:
            A `CoreConfig` object created from config.
        