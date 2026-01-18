from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import BadgeRenderMixin, CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectBadge(SaveMixin, ObjectDeleteMixin, RESTObject):
    pass