from __future__ import annotations as _annotations
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple, Union, cast
from dotenv import dotenv_values
from pydantic import AliasChoices, AliasPath, BaseModel, Json
from pydantic._internal._typing_extra import WithArgsTypes, origin_is_union
from pydantic._internal._utils import deep_update, is_model_class, lenient_issubclass
from pydantic.fields import FieldInfo
from typing_extensions import get_args, get_origin
from pydantic_settings.utils import path_type_label
class TomlConfigSettingsSource(InitSettingsSource, ConfigFileSourceMixin):
    """
    A source class that loads variables from a JSON file
    """

    def __init__(self, settings_cls: type[BaseSettings], toml_file: PathType | None=DEFAULT_PATH):
        self.toml_file_path = toml_file if toml_file != DEFAULT_PATH else settings_cls.model_config.get('toml_file')
        self.toml_data = self._read_files(self.toml_file_path)
        super().__init__(settings_cls, self.toml_data)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        import_toml()
        with open(file_path, mode='rb') as toml_file:
            if sys.version_info < (3, 11):
                return tomli.load(toml_file)
            return tomllib.load(toml_file)