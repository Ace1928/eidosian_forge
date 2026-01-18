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
class UserImpersonationTokenManager(NoUpdateMixin, RESTManager):
    _path = '/users/{user_id}/impersonation_tokens'
    _obj_cls = UserImpersonationToken
    _from_parent_attrs = {'user_id': 'id'}
    _create_attrs = RequiredOptional(required=('name', 'scopes'), optional=('expires_at',))
    _list_filters = ('state',)
    _types = {'scopes': ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> UserImpersonationToken:
        return cast(UserImpersonationToken, super().get(id=id, lazy=lazy, **kwargs))