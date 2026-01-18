from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectMergeRequestAwardEmoji(ObjectDeleteMixin, RESTObject):
    pass