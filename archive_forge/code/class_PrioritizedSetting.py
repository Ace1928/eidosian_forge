from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
class PrioritizedSetting(Generic[T]):
    """ Return a value for a global setting according to configuration precedence.

    The following methods are searched in order for the setting:

    7. immediately supplied values
    6. previously user-set values (e.g. set from command line)
    5. user-specified config override file
    4. environment variable
    3. local user config file
    2. global system config file (not yet implemented)
    1. local defaults
    0. global defaults

    Ref: https://stackoverflow.com/a/11077282/3406693

    If a value cannot be determined, a ValueError is raised.

    The ``env_var`` argument specifies the name of an environment to check for
    setting values, e.g. ``"BOKEH_LOG_LEVEL"``.

    The optional ``default`` argument specified an implicit default value for
    the setting that is returned if no other methods provide a value.

    A ``convert`` agument may be provided to convert values before they are
    returned. For instance to concert log levels in environment variables
    to ``logging`` module values.
    """
    _parent: Settings | None
    _user_value: Unset[str | T]

    def __init__(self, name: str, env_var: str | None=None, default: Unset[T]=_Unset, dev_default: Unset[T]=_Unset, convert: Callable[[T | str], T] | None=None, help: str='') -> None:
        self._convert = convert if convert else convert_str
        self._default = default
        self._dev_default = dev_default
        self._env_var = env_var
        self._help = help
        self._name = name
        self._parent = None
        self._user_value = _Unset

    def __call__(self, value: T | str | None=None, default: Unset[T]=_Unset) -> T:
        """Return the setting value according to the standard precedence.

        Args:
            value (any, optional):
                An optional immediate value. If not None, the value will
                be converted, then returned.

            default (any, optional):
                An optional default value that only takes precendence over
                implicit default values specified on the property itself.

        Returns:
            str or int or float

        Raises:
            RuntimeError
        """
        if value is not None:
            return self._convert(value)
        if self._user_value is not _Unset:
            return self._convert(self._user_value)
        if self._parent and self._name in self._parent.config_override:
            return self._convert(self._parent.config_override[self._name])
        if self._env_var and self._env_var in os.environ:
            return self._convert(os.environ[self._env_var])
        if self._parent and self._name in self._parent.config_user:
            return self._convert(self._parent.config_user[self._name])
        if self._parent and self._name in self._parent.config_system:
            return self._convert(self._parent.config_system[self._name])
        if is_dev() and self._dev_default is not _Unset:
            return self._convert(self._dev_default)
        if default is not _Unset:
            return self._convert(default)
        if self._default is not _Unset:
            return self._convert(self._default)
        raise RuntimeError(f'No configured value found for setting {self._name!r}')

    def __get__(self, instance: Any, owner: type[Any]) -> PrioritizedSetting[T]:
        return self

    def __set__(self, instance: Any, value: str | T) -> None:
        self.set_value(value)

    def __delete__(self, instance: Any) -> None:
        self.unset_value()

    def set_value(self, value: str | T) -> None:
        """ Specify a value for this setting programmatically.

        A value set this way takes precedence over all other methods except
        immediate values.

        Args:
            value (str or int or float):
                A user-set value for this setting

        Returns:
            None
        """
        self._user_value = value

    def unset_value(self) -> None:
        """ Unset the previous user value such that the priority is reset.

        """
        self._user_value = _Unset

    @property
    def env_var(self) -> str | None:
        return self._env_var

    @property
    def default(self) -> Unset[T]:
        return self._default

    @property
    def dev_default(self) -> Unset[T]:
        return self._dev_default

    @property
    def name(self) -> str:
        return self._name

    @property
    def help(self) -> str:
        return self._help

    @property
    def convert_type(self) -> str:
        if self._convert is convert_str:
            return 'String'
        if self._convert is convert_bool:
            return 'Bool'
        if self._convert is convert_int:
            return 'Int'
        if self._convert is convert_logging:
            return 'Log Level'
        if self._convert is convert_str_seq:
            return 'List[String]'
        if self._convert is convert_validation:
            return 'Validation Level'
        if self._convert is convert_ico_path:
            return 'Ico Path'
        raise RuntimeError('unreachable')