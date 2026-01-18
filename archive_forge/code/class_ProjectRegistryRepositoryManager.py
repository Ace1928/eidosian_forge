from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectRegistryRepositoryManager(DeleteMixin, ListMixin, RESTManager):
    _path = '/projects/{project_id}/registry/repositories'
    _obj_cls = ProjectRegistryRepository
    _from_parent_attrs = {'project_id': 'id'}