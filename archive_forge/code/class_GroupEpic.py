from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .events import GroupEpicResourceLabelEventManager  # noqa: F401
from .notes import GroupEpicNoteManager  # noqa: F401
class GroupEpic(ObjectDeleteMixin, SaveMixin, RESTObject):
    _id_attr = 'iid'
    issues: 'GroupEpicIssueManager'
    resourcelabelevents: GroupEpicResourceLabelEventManager
    notes: GroupEpicNoteManager