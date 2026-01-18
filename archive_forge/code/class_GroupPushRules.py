from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupPushRules(SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = None