import json
import logging
import os
import urllib
from typing import TYPE_CHECKING, Any, Dict, Optional
import requests
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.launch.utils import LAUNCH_DEFAULT_PROJECT
from wandb.sdk.lib import retry, runid
from wandb.sdk.lib.gql_request import GraphQLSession
class RetryingClient:
    INFO_QUERY = gql('\n        query ServerInfo{\n            serverInfo {\n                cliVersionInfo\n                latestLocalVersionInfo {\n                    outOfDate\n                    latestVersionString\n                    versionOnThisInstanceString\n                }\n            }\n        }\n        ')

    def __init__(self, client: Client):
        self._server_info = None
        self._client = client

    @property
    def app_url(self):
        return util.app_url(self._client.transport.url.replace('/graphql', '')) + '/'

    @retry.retriable(retry_timedelta=RETRY_TIMEDELTA, check_retry_fn=util.no_retry_auth, retryable_exceptions=(RetryError, requests.RequestException))
    def execute(self, *args, **kwargs):
        try:
            return self._client.execute(*args, **kwargs)
        except requests.exceptions.ReadTimeout:
            if 'timeout' not in kwargs:
                timeout = self._client.transport.default_timeout
                wandb.termwarn(f'A graphql request initiated by the public wandb API timed out (timeout={timeout} sec). Create a new API with an integer timeout larger than {timeout}, e.g., `api = wandb.Api(timeout={timeout + 10})` to increase the graphql timeout.')
            raise

    @property
    def server_info(self):
        if self._server_info is None:
            self._server_info = self.execute(self.INFO_QUERY).get('serverInfo')
        return self._server_info

    def version_supported(self, min_version):
        from wandb.util import parse_version
        return parse_version(min_version) <= parse_version(self.server_info['cliVersionInfo']['max_cli_version'])