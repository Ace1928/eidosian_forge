from typing import Any, cast, List, Optional, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupRunnerManager(ListMixin, RESTManager):
    _path = '/groups/{group_id}/runners'
    _obj_cls = GroupRunner
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('runner_id',))
    _list_filters = ('scope', 'tag_list')
    _types = {'tag_list': types.CommaSeparatedListAttribute}