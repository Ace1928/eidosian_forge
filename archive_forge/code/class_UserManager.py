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
class UserManager(CRUDMixin, RESTManager):
    _path = '/users'
    _obj_cls = User
    _list_filters = ('active', 'blocked', 'username', 'extern_uid', 'provider', 'external', 'search', 'custom_attributes', 'status', 'two_factor')
    _create_attrs = RequiredOptional(optional=('email', 'username', 'name', 'password', 'reset_password', 'skype', 'linkedin', 'twitter', 'projects_limit', 'extern_uid', 'provider', 'bio', 'admin', 'can_create_group', 'website_url', 'skip_confirmation', 'external', 'organization', 'location', 'avatar', 'public_email', 'private_profile', 'color_scheme_id', 'theme_id'))
    _update_attrs = RequiredOptional(required=('email', 'username', 'name'), optional=('password', 'skype', 'linkedin', 'twitter', 'projects_limit', 'extern_uid', 'provider', 'bio', 'admin', 'can_create_group', 'website_url', 'skip_reconfirmation', 'external', 'organization', 'location', 'avatar', 'public_email', 'private_profile', 'color_scheme_id', 'theme_id'))
    _types = {'confirm': types.LowercaseStringAttribute, 'avatar': types.ImageAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> User:
        return cast(User, super().get(id=id, lazy=lazy, **kwargs))