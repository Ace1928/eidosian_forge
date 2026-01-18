from typing import Any, cast, List, Union
from gitlab import cli
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectServiceManager(ProjectIntegrationManager):
    _obj_cls = ProjectService

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectService:
        return cast(ProjectService, super().get(id=id, lazy=lazy, **kwargs))