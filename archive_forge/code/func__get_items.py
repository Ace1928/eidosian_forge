import json
import os
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.errors import CommError
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.data_types._dtypes import InvalidType, Type, TypeRegistry
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
@normalize_exceptions
def _get_items(self):
    query = gql('\n            query GetRunQueueItems($projectName: String!, $entityName: String!, $runQueue: String!) {\n                project(name: $projectName, entityName: $entityName) {\n                    runQueue(name: $runQueue) {\n                        runQueueItems(first: 100) {\n                            edges {\n                                node {\n                                    id\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        ')
    variable_values = {'projectName': LAUNCH_DEFAULT_PROJECT, 'entityName': self._entity, 'runQueue': self._name}
    res = self._client.execute(query, variable_values)
    self._items = []
    for item in res['project']['runQueue']['runQueueItems']['edges']:
        self._items.append(QueuedRun(self._client, self._entity, LAUNCH_DEFAULT_PROJECT, self._name, item['node']['id']))