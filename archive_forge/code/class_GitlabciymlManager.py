from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
class GitlabciymlManager(RetrieveMixin, RESTManager):
    _path = '/templates/gitlab_ci_ymls'
    _obj_cls = Gitlabciyml

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Gitlabciyml:
        return cast(Gitlabciyml, super().get(id=id, lazy=lazy, **kwargs))