from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class GroupRegistryRepositoryManager(ListMixin, RESTManager):
    _path = '/groups/{group_id}/registry/repositories'
    _obj_cls = ProjectRegistryRepository
    _from_parent_attrs = {'group_id': 'id'}