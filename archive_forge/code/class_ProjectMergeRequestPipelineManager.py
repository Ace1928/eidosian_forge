from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectMergeRequestPipelineManager(CreateMixin, ListMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/pipelines'
    _obj_cls = ProjectMergeRequestPipeline
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}