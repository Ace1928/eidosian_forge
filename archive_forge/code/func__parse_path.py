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
def _parse_path(self, path):
    """Parse url, filepath, or docker paths.

        Allows paths in the following formats:
        - url: entity/project/runs/id
        - path: entity/project/id
        - docker: entity/project:id

        Entity is optional and will fall back to the current logged-in user.
        """
    project = self.settings['project'] or 'uncategorized'
    entity = self.settings['entity'] or self.default_entity
    parts = path.replace('/runs/', '/').replace('/sweeps/', '/').strip('/ ').split('/')
    if ':' in parts[-1]:
        id = parts[-1].split(':')[-1]
        parts[-1] = parts[-1].split(':')[0]
    elif parts[-1]:
        id = parts[-1]
    if len(parts) == 1 and project != 'uncategorized':
        pass
    elif len(parts) > 1:
        project = parts[1]
        if entity and id == project:
            project = parts[0]
        else:
            entity = parts[0]
        if len(parts) == 3:
            entity = parts[0]
    else:
        project = parts[0]
    return (entity, project, id)