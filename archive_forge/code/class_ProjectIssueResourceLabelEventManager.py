from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class ProjectIssueResourceLabelEventManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/resource_label_events'
    _obj_cls = ProjectIssueResourceLabelEvent
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueResourceLabelEvent:
        return cast(ProjectIssueResourceLabelEvent, super().get(id=id, lazy=lazy, **kwargs))