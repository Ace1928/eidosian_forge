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
def _get_default_resource_config(self):
    query = gql('\n            query GetDefaultResourceConfig($entityName: String!, $id: ID!) {\n                entity(name: $entityName) {\n                    defaultResourceConfig(id: $id) {\n                        config\n                        resource\n                        templateVariables {\n                            name\n                            schema\n                        }\n                    }\n                }\n            }\n        ')
    variable_values = {'entityName': self._entity, 'id': self._default_resource_config_id}
    res = self._client.execute(query, variable_values)
    self._type = res['entity']['defaultResourceConfig']['resource']
    self._default_resource_config = res['entity']['defaultResourceConfig']['config']
    self._template_variables = res['entity']['defaultResourceConfig']['templateVariables']