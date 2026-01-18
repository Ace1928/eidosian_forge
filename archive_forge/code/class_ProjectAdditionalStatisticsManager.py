from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, RefreshMixin
from gitlab.types import ArrayAttribute
class ProjectAdditionalStatisticsManager(GetWithoutIdMixin, RESTManager):
    _path = '/projects/{project_id}/statistics'
    _obj_cls = ProjectAdditionalStatistics
    _from_parent_attrs = {'project_id': 'id'}

    def get(self, **kwargs: Any) -> ProjectAdditionalStatistics:
        return cast(ProjectAdditionalStatistics, super().get(**kwargs))