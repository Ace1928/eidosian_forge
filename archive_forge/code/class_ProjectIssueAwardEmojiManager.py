from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectIssueAwardEmojiManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/award_emoji'
    _obj_cls = ProjectIssueAwardEmoji
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueAwardEmoji:
        return cast(ProjectIssueAwardEmoji, super().get(id=id, lazy=lazy, **kwargs))