from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
class GroupAuditEventManager(RetrieveMixin, RESTManager):
    _path = '/groups/{group_id}/audit_events'
    _obj_cls = GroupAuditEvent
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('created_after', 'created_before')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupAuditEvent:
        return cast(GroupAuditEvent, super().get(id=id, lazy=lazy, **kwargs))