from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
class GroupNotificationSettings(NotificationSettings):
    pass