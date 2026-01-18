from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectPipelineBridgeManager(ListMixin, RESTManager):
    _path = '/projects/{project_id}/pipelines/{pipeline_id}/bridges'
    _obj_cls = ProjectPipelineBridge
    _from_parent_attrs = {'project_id': 'project_id', 'pipeline_id': 'id'}
    _list_filters = ('scope',)