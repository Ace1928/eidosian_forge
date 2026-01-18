from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
class GroupNotificationSettingsManager(NotificationSettingsManager):
    _path = '/groups/{group_id}/notification_settings'
    _obj_cls = GroupNotificationSettings
    _from_parent_attrs = {'group_id': 'id'}

    def get(self, **kwargs: Any) -> GroupNotificationSettings:
        return cast(GroupNotificationSettings, super().get(id=id, **kwargs))