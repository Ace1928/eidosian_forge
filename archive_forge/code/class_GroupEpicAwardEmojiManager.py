from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class GroupEpicAwardEmojiManager(NoUpdateMixin, RESTManager):
    _path = '/groups/{group_id}/epics/{epic_iid}/award_emoji'
    _obj_cls = GroupEpicAwardEmoji
    _from_parent_attrs = {'group_id': 'group_id', 'epic_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupEpicAwardEmoji:
        return cast(GroupEpicAwardEmoji, super().get(id=id, lazy=lazy, **kwargs))