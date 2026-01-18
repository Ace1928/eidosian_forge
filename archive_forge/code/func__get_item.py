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
def _get_item(self):
    query = gql('\n            query GetRunQueueItem($projectName: String!, $entityName: String!, $runQueue: String!, $itemId: ID!) {\n                project(name: $projectName, entityName: $entityName) {\n                    runQueue(name: $runQueue) {\n                        runQueueItem(id: $itemId) {\n                            id\n                            state\n                            associatedRunId\n                        }\n                    }\n                }\n            }\n        ')
    variable_values = {'projectName': self.project_queue, 'entityName': self._entity, 'runQueue': self.queue_name, 'itemId': self.id}
    try:
        res = self.client.execute(query, variable_values)
        if res['project']['runQueue'].get('runQueueItem') is not None:
            return res['project']['runQueue']['runQueueItem']
    except Exception as e:
        if 'Cannot query field' not in str(e):
            raise LaunchError(f'Unknown exception: {e}')
    return self._get_run_queue_item_legacy()