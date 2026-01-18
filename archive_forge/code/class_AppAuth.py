import abc
import base64
import time
from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional, Union
import jwt
from requests import utils
from github import Consts
from github.InstallationAuthorization import InstallationAuthorization
from github.Requester import Requester, WithRequester
class AppAuth(JWT):
    """
    This class is used to authenticate as a GitHub App.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app
    """

    def __init__(self, app_id: Union[int, str], private_key: str, jwt_expiry: int=Consts.DEFAULT_JWT_EXPIRY, jwt_issued_at: int=Consts.DEFAULT_JWT_ISSUED_AT, jwt_algorithm: str=Consts.DEFAULT_JWT_ALGORITHM):
        assert isinstance(app_id, (int, str)), app_id
        if isinstance(app_id, str):
            assert len(app_id) > 0, 'app_id must not be empty'
        assert isinstance(private_key, str)
        assert len(private_key) > 0, 'private_key must not be empty'
        assert isinstance(jwt_expiry, int), jwt_expiry
        assert Consts.MIN_JWT_EXPIRY <= jwt_expiry <= Consts.MAX_JWT_EXPIRY, jwt_expiry
        self._app_id = app_id
        self._private_key = private_key
        self._jwt_expiry = jwt_expiry
        self._jwt_issued_at = jwt_issued_at
        self._jwt_algorithm = jwt_algorithm

    @property
    def app_id(self) -> Union[int, str]:
        return self._app_id

    @property
    def private_key(self) -> str:
        return self._private_key

    @property
    def token(self) -> str:
        return self.create_jwt()

    def get_installation_auth(self, installation_id: int, token_permissions: Optional[Dict[str, str]]=None, requester: Optional[Requester]=None) -> 'AppInstallationAuth':
        """
        Creates a github.Auth.AppInstallationAuth instance for an installation.
        :param installation_id: installation id
        :param token_permissions: optional permissions
        :param requester: optional requester with app authentication
        :return:
        """
        return AppInstallationAuth(self, installation_id, token_permissions, requester)

    def create_jwt(self, expiration: Optional[int]=None) -> str:
        """
        Create a signed JWT
        https://docs.github.com/en/developers/apps/building-github-apps/authenticating-with-github-apps#authenticating-as-a-github-app

        :return string: jwt
        """
        if expiration is not None:
            assert isinstance(expiration, int), expiration
            assert Consts.MIN_JWT_EXPIRY <= expiration <= Consts.MAX_JWT_EXPIRY, expiration
        now = int(time.time())
        payload = {'iat': now + self._jwt_issued_at, 'exp': now + (expiration if expiration is not None else self._jwt_expiry), 'iss': self._app_id}
        encrypted = jwt.encode(payload, key=self.private_key, algorithm=self._jwt_algorithm)
        if isinstance(encrypted, bytes):
            return encrypted.decode('utf-8')
        return encrypted