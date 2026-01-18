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
class DotEnvSettingsSource(EnvSettingsSource):
    """
    Source class for loading settings values from env files.
    """

    def __init__(self, settings_cls: type[BaseSettings], env_file: DotenvType | None=ENV_FILE_SENTINEL, env_file_encoding: str | None=None, case_sensitive: bool | None=None, env_prefix: str | None=None, env_nested_delimiter: str | None=None, env_ignore_empty: bool | None=None, env_parse_none_str: str | None=None) -> None:
        self.env_file = env_file if env_file != ENV_FILE_SENTINEL else settings_cls.model_config.get('env_file')
        self.env_file_encoding = env_file_encoding if env_file_encoding is not None else settings_cls.model_config.get('env_file_encoding')
        super().__init__(settings_cls, case_sensitive, env_prefix, env_nested_delimiter, env_ignore_empty, env_parse_none_str)

    def _load_env_vars(self) -> Mapping[str, str | None]:
        return self._read_env_files()

    def _read_env_files(self) -> Mapping[str, str | None]:
        env_files = self.env_file
        if env_files is None:
            return {}
        if isinstance(env_files, (str, os.PathLike)):
            env_files = [env_files]
        dotenv_vars: dict[str, str | None] = {}
        for env_file in env_files:
            env_path = Path(env_file).expanduser()
            if env_path.is_file():
                dotenv_vars.update(read_env_file(env_path, encoding=self.env_file_encoding, case_sensitive=self.case_sensitive, ignore_empty=self.env_ignore_empty, parse_none_str=self.env_parse_none_str))
        return dotenv_vars

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = super().__call__()
        is_extra_allowed = self.config.get('extra') != 'forbid'
        for env_name, env_value in self.env_vars.items():
            if not env_value:
                continue
            env_used = False
            for field_name, field in self.settings_cls.model_fields.items():
                for _, field_env_name, _ in self._extract_field_info(field, field_name):
                    if env_name.startswith(field_env_name):
                        env_used = True
                        break
            if not env_used:
                if is_extra_allowed and env_name.startswith(self.env_prefix):
                    normalized_env_name = env_name[len(self.env_prefix):]
                    data[normalized_env_name] = env_value
                else:
                    data[env_name] = env_value
        return data

    def __repr__(self) -> str:
        return f'DotEnvSettingsSource(env_file={self.env_file!r}, env_file_encoding={self.env_file_encoding!r}, env_nested_delimiter={self.env_nested_delimiter!r}, env_prefix_len={self.env_prefix_len!r})'