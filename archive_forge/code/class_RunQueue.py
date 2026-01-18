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
class RunQueue:

    def __init__(self, client: 'RetryingClient', name: str, entity: str, prioritization_mode: Optional[RunQueuePrioritizationMode]=None, _access: Optional[RunQueueAccessType]=None, _default_resource_config_id: Optional[int]=None, _default_resource_config: Optional[dict]=None) -> None:
        self._name: str = name
        self._client = client
        self._entity = entity
        self._prioritization_mode = prioritization_mode
        self._access = _access
        self._default_resource_config_id = _default_resource_config_id
        self._default_resource_config = _default_resource_config
        self._template_variables = None
        self._type = None
        self._items = None
        self._id = None

    @property
    def name(self):
        return self._name

    @property
    def entity(self):
        return self._entity

    @property
    def prioritization_mode(self) -> RunQueuePrioritizationMode:
        if self._prioritization_mode is None:
            self._get_metadata()
        return self._prioritization_mode

    @property
    def access(self) -> RunQueueAccessType:
        if self._access is None:
            self._get_metadata()
        return self._access

    @property
    def type(self) -> RunQueueResourceType:
        if self._type is None:
            if self._default_resource_config_id is None:
                self._get_metadata()
            self._get_default_resource_config()
        return self._type

    @property
    def default_resource_config(self):
        if self._default_resource_config is None:
            if self._default_resource_config_id is None:
                self._get_metadata()
            self._get_default_resource_config()
        return self._default_resource_config

    @property
    def template_variables(self):
        if self._template_variables is None:
            if self._default_resource_config_id is None:
                self._get_metadata()
            self._get_default_resource_config()
        return self._template_variables

    @property
    def id(self) -> str:
        if self._id is None:
            self._get_metadata()
        return self._id

    @property
    def items(self) -> List[QueuedRun]:
        """Up to the first 100 queued runs. Modifying this list will not modify the queue or any enqueued items!"""
        if self._items is None:
            self._get_items()
        return self._items

    @normalize_exceptions
    def delete(self):
        """Delete the run queue from the wandb backend."""
        query = gql('\n            mutation DeleteRunQueue($id: ID!) {\n                deleteRunQueues(input: {queueIDs: [$id]}) {\n                    success\n                    clientMutationId\n                }\n            }\n            ')
        variable_values = {'id': self.id}
        res = self._client.execute(query, variable_values)
        if res['deleteRunQueues']['success']:
            self._id = None
            self._access = None
            self._default_resource_config_id = None
            self._default_resource_config = None
            self._items = None
        else:
            raise CommError(f'Failed to delete run queue {self.name}')

    def __repr__(self):
        return f'<RunQueue {self._entity}/{self._name}>'

    @normalize_exceptions
    def _get_metadata(self):
        query = gql('\n            query GetRunQueueMetadata($projectName: String!, $entityName: String!, $runQueue: String!) {\n                project(name: $projectName, entityName: $entityName) {\n                    runQueue(name: $runQueue) {\n                        id\n                        access\n                        defaultResourceConfigID\n                        prioritizationMode\n                    }\n                }\n            }\n        ')
        variable_values = {'projectName': LAUNCH_DEFAULT_PROJECT, 'entityName': self._entity, 'runQueue': self._name}
        res = self._client.execute(query, variable_values)
        self._id = res['project']['runQueue']['id']
        self._access = res['project']['runQueue']['access']
        self._default_resource_config_id = res['project']['runQueue']['defaultResourceConfigID']
        if self._default_resource_config_id is None:
            self._default_resource_config = {}
        self._prioritization_mode = res['project']['runQueue']['prioritizationMode']

    @normalize_exceptions
    def _get_default_resource_config(self):
        query = gql('\n            query GetDefaultResourceConfig($entityName: String!, $id: ID!) {\n                entity(name: $entityName) {\n                    defaultResourceConfig(id: $id) {\n                        config\n                        resource\n                        templateVariables {\n                            name\n                            schema\n                        }\n                    }\n                }\n            }\n        ')
        variable_values = {'entityName': self._entity, 'id': self._default_resource_config_id}
        res = self._client.execute(query, variable_values)
        self._type = res['entity']['defaultResourceConfig']['resource']
        self._default_resource_config = res['entity']['defaultResourceConfig']['config']
        self._template_variables = res['entity']['defaultResourceConfig']['templateVariables']

    @normalize_exceptions
    def _get_items(self):
        query = gql('\n            query GetRunQueueItems($projectName: String!, $entityName: String!, $runQueue: String!) {\n                project(name: $projectName, entityName: $entityName) {\n                    runQueue(name: $runQueue) {\n                        runQueueItems(first: 100) {\n                            edges {\n                                node {\n                                    id\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        ')
        variable_values = {'projectName': LAUNCH_DEFAULT_PROJECT, 'entityName': self._entity, 'runQueue': self._name}
        res = self._client.execute(query, variable_values)
        self._items = []
        for item in res['project']['runQueue']['runQueueItems']['edges']:
            self._items.append(QueuedRun(self._client, self._entity, LAUNCH_DEFAULT_PROJECT, self._name, item['node']['id']))

    @classmethod
    def create(cls, name: str, resource: 'RunQueueResourceType', entity: Optional[str]=None, prioritization_mode: Optional['RunQueuePrioritizationMode']=None, config: Optional[dict]=None, template_variables: Optional[dict]=None) -> 'RunQueue':
        public_api = Api()
        return public_api.create_run_queue(name, resource, entity, prioritization_mode, config, template_variables)