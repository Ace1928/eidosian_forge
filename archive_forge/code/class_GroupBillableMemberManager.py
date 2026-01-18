from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupBillableMemberManager(ListMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/billable_members'
    _obj_cls = GroupBillableMember
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('search', 'sort')