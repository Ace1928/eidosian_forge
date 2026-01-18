from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectPipelineSchedulePipelineManager(ListMixin, RESTManager):
    _path = '/projects/{project_id}/pipeline_schedules/{pipeline_schedule_id}/pipelines'
    _obj_cls = ProjectPipelineSchedulePipeline
    _from_parent_attrs = {'project_id': 'project_id', 'pipeline_schedule_id': 'id'}