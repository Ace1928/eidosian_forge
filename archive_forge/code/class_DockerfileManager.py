from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
class DockerfileManager(RetrieveMixin, RESTManager):
    _path = '/templates/dockerfiles'
    _obj_cls = Dockerfile

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Dockerfile:
        return cast(Dockerfile, super().get(id=id, lazy=lazy, **kwargs))