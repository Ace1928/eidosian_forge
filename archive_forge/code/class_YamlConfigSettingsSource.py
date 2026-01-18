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
class YamlConfigSettingsSource(InitSettingsSource, ConfigFileSourceMixin):
    """
    A source class that loads variables from a yaml file
    """

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: PathType | None=DEFAULT_PATH, yaml_file_encoding: str | None=None):
        self.yaml_file_path = yaml_file if yaml_file != DEFAULT_PATH else settings_cls.model_config.get('yaml_file')
        self.yaml_file_encoding = yaml_file_encoding if yaml_file_encoding is not None else settings_cls.model_config.get('yaml_file_encoding')
        self.yaml_data = self._read_files(self.yaml_file_path)
        super().__init__(settings_cls, self.yaml_data)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        import_yaml()
        with open(file_path, encoding=self.yaml_file_encoding) as yaml_file:
            return yaml.safe_load(yaml_file)