from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ListMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectPagesDomain(SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = 'domain'