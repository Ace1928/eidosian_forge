from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectPipelineScheduleManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/pipeline_schedules'
    _obj_cls = ProjectPipelineSchedule
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('description', 'ref', 'cron'), optional=('cron_timezone', 'active'))
    _update_attrs = RequiredOptional(optional=('description', 'ref', 'cron', 'cron_timezone', 'active'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectPipelineSchedule:
        return cast(ProjectPipelineSchedule, super().get(id=id, lazy=lazy, **kwargs))