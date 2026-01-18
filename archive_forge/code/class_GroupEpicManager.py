from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .events import GroupEpicResourceLabelEventManager  # noqa: F401
from .notes import GroupEpicNoteManager  # noqa: F401
class GroupEpicManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/epics'
    _obj_cls = GroupEpic
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('author_id', 'labels', 'order_by', 'sort', 'search')
    _create_attrs = RequiredOptional(required=('title',), optional=('labels', 'description', 'start_date', 'end_date'))
    _update_attrs = RequiredOptional(optional=('title', 'labels', 'description', 'start_date', 'end_date'))
    _types = {'labels': types.CommaSeparatedListAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupEpic:
        return cast(GroupEpic, super().get(id=id, lazy=lazy, **kwargs))