from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class RegistryRepositoryManager(GetMixin, RESTManager):
    _path = '/registry/repositories'
    _obj_cls = RegistryRepository

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> RegistryRepository:
        return cast(RegistryRepository, super().get(id=id, lazy=lazy, **kwargs))