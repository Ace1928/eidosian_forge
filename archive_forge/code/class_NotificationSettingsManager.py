from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
class NotificationSettingsManager(GetWithoutIdMixin, UpdateMixin, RESTManager):
    _path = '/notification_settings'
    _obj_cls = NotificationSettings
    _update_attrs = RequiredOptional(optional=('level', 'notification_email', 'new_note', 'new_issue', 'reopen_issue', 'close_issue', 'reassign_issue', 'new_merge_request', 'reopen_merge_request', 'close_merge_request', 'reassign_merge_request', 'merge_merge_request'))

    def get(self, **kwargs: Any) -> NotificationSettings:
        return cast(NotificationSettings, super().get(**kwargs))