from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class GroupEpicNoteAwardEmojiManager(NoUpdateMixin, RESTManager):
    _path = '/groups/{group_id}/epics/{epic_iid}/notes/{note_id}/award_emoji'
    _obj_cls = GroupEpicNoteAwardEmoji
    _from_parent_attrs = {'group_id': 'group_id', 'epic_iid': 'epic_iid', 'note_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupEpicNoteAwardEmoji:
        return cast(GroupEpicNoteAwardEmoji, super().get(id=id, lazy=lazy, **kwargs))