from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
@dataclasses.dataclass(frozen=True)
class LegacyHostOptions:
    """Legacy host options used prior to the availability of separate controller and target host configuration."""
    python: t.Optional[str] = None
    python_interpreter: t.Optional[str] = None
    local: t.Optional[bool] = None
    venv: t.Optional[bool] = None
    venv_system_site_packages: t.Optional[bool] = None
    remote: t.Optional[str] = None
    remote_provider: t.Optional[str] = None
    remote_arch: t.Optional[str] = None
    docker: t.Optional[str] = None
    docker_privileged: t.Optional[bool] = None
    docker_seccomp: t.Optional[str] = None
    docker_memory: t.Optional[int] = None
    windows: t.Optional[list[str]] = None
    platform: t.Optional[list[str]] = None
    platform_collection: t.Optional[list[tuple[str, str]]] = None
    platform_connection: t.Optional[list[tuple[str, str]]] = None
    inventory: t.Optional[str] = None

    @staticmethod
    def create(namespace: t.Union[argparse.Namespace, types.SimpleNamespace]) -> LegacyHostOptions:
        """Create legacy host options from the given namespace."""
        kwargs = {field.name: getattr(namespace, field.name, None) for field in dataclasses.fields(LegacyHostOptions)}
        if kwargs['python'] == 'default':
            kwargs['python'] = None
        return LegacyHostOptions(**kwargs)

    @staticmethod
    def purge_namespace(namespace: t.Union[argparse.Namespace, types.SimpleNamespace]) -> None:
        """Purge legacy host options fields from the given namespace."""
        for field in dataclasses.fields(LegacyHostOptions):
            if hasattr(namespace, field.name):
                delattr(namespace, field.name)

    @staticmethod
    def purge_args(args: list[str]) -> list[str]:
        """Purge legacy host options from the given command line arguments."""
        fields: tuple[dataclasses.Field, ...] = dataclasses.fields(LegacyHostOptions)
        filters: dict[str, int] = {get_option_name(field.name): 0 if field.type is t.Optional[bool] else 1 for field in fields}
        return filter_args(args, filters)

    def get_options_used(self) -> tuple[str, ...]:
        """Return a tuple of the command line options used."""
        fields: tuple[dataclasses.Field, ...] = dataclasses.fields(self)
        options = tuple(sorted((get_option_name(field.name) for field in fields if getattr(self, field.name))))
        return options