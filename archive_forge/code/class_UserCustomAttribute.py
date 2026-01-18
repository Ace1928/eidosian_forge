from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, ObjectDeleteMixin, RetrieveMixin, SetMixin
class UserCustomAttribute(ObjectDeleteMixin, RESTObject):
    _id_attr = 'key'