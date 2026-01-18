from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
class CurrentUserEmailManager(RetrieveMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/user/emails'
    _obj_cls = CurrentUserEmail
    _create_attrs = RequiredOptional(required=('email',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> CurrentUserEmail:
        return cast(CurrentUserEmail, super().get(id=id, lazy=lazy, **kwargs))