from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class GroupBoard(SaveMixin, ObjectDeleteMixin, RESTObject):
    lists: GroupBoardListManager