from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
class ProjectAuditEventManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/audit_events'
    _obj_cls = ProjectAuditEvent
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('created_after', 'created_before')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectAuditEvent:
        return cast(ProjectAuditEvent, super().get(id=id, lazy=lazy, **kwargs))