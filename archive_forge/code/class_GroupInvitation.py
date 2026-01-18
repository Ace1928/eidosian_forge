from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.exceptions import GitlabInvitationError
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import ArrayAttribute, CommaSeparatedListAttribute, RequiredOptional
class GroupInvitation(SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = 'email'