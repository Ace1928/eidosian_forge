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
class AppInstallationAuth(Auth, WithRequester['AppInstallationAuth']):
    """
    This class is used to authenticate as a GitHub App Installation.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation
    """
    __integration: Optional['GithubIntegration'] = None
    __installation_authorization: Optional[InstallationAuthorization] = None

    def __init__(self, app_auth: AppAuth, installation_id: int, token_permissions: Optional[Dict[str, str]]=None, requester: Optional[Requester]=None):
        super().__init__()
        assert isinstance(app_auth, AppAuth), app_auth
        assert isinstance(installation_id, int), installation_id
        assert token_permissions is None or isinstance(token_permissions, dict), token_permissions
        self._app_auth = app_auth
        self._installation_id = installation_id
        self._token_permissions = token_permissions
        if requester is not None:
            self.withRequester(requester)

    def withRequester(self, requester: Requester) -> 'AppInstallationAuth':
        super().withRequester(requester.withAuth(self._app_auth))
        from github.GithubIntegration import GithubIntegration
        self.__integration = GithubIntegration(**self.requester.kwargs)
        return self

    @property
    def app_id(self) -> Union[int, str]:
        return self._app_auth.app_id

    @property
    def private_key(self) -> str:
        return self._app_auth.private_key

    @property
    def installation_id(self) -> int:
        return self._installation_id

    @property
    def token_permissions(self) -> Optional[Dict[str, str]]:
        return self._token_permissions

    @property
    def token_type(self) -> str:
        return 'token'

    @property
    def token(self) -> str:
        if self.__installation_authorization is None or self._is_expired:
            self.__installation_authorization = self._get_installation_authorization()
        return self.__installation_authorization.token

    @property
    def _is_expired(self) -> bool:
        assert self.__installation_authorization is not None
        token_expires_at = self.__installation_authorization.expires_at - TOKEN_REFRESH_THRESHOLD_TIMEDELTA
        return token_expires_at < datetime.now(timezone.utc)

    def _get_installation_authorization(self) -> InstallationAuthorization:
        assert self.__integration is not None, 'Method withRequester(Requester) must be called first'
        return self.__integration.get_access_token(self._installation_id, permissions=self._token_permissions)