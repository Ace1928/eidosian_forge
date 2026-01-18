from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectMergeRequestNoteAwardEmojiManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/notes/{note_id}/award_emoji'
    _obj_cls = ProjectMergeRequestNoteAwardEmoji
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'mr_iid', 'note_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestNoteAwardEmoji:
        return cast(ProjectMergeRequestNoteAwardEmoji, super().get(id=id, lazy=lazy, **kwargs))