import ast
import asyncio
import base64
import datetime
import functools
import http.client
import json
import logging
import os
import re
import socket
import sys
import threading
from copy import deepcopy
from typing import (
import click
import requests
import yaml
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis.normalize import normalize_exceptions, parse_backend_error_messages
from wandb.errors import CommError, UnsupportedError, UsageError
from wandb.integration.sagemaker import parse_sm_secrets
from wandb.old.settings import Settings
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.gql_request import GraphQLSession
from wandb.sdk.lib.hashutil import B64MD5, md5_file_b64
from ..lib import retry
from ..lib.filenames import DIFF_FNAME, METADATA_FNAME
from ..lib.gitlib import GitRepo
from . import context
from .progress import AsyncProgress, Progress
def _get_create_artifact_mutation(self, fields: List, history_step: Optional[int], distributed_id: Optional[str]) -> str:
    types = ''
    values = ''
    if 'historyStep' in fields and history_step not in [0, None]:
        types += '$historyStep: Int64!,'
        values += 'historyStep: $historyStep,'
    if distributed_id:
        types += '$distributedID: String,'
        values += 'distributedID: $distributedID,'
    if 'clientID' in fields:
        types += '$clientID: ID,'
        values += 'clientID: $clientID,'
    if 'sequenceClientID' in fields:
        types += '$sequenceClientID: ID,'
        values += 'sequenceClientID: $sequenceClientID,'
    if 'enableDigestDeduplication' in fields:
        values += 'enableDigestDeduplication: true,'
    if 'ttlDurationSeconds' in fields:
        types += '$ttlDurationSeconds: Int64,'
        values += 'ttlDurationSeconds: $ttlDurationSeconds,'
    query_template = '\n            mutation CreateArtifact(\n                $artifactTypeName: String!,\n                $artifactCollectionNames: [String!],\n                $entityName: String!,\n                $projectName: String!,\n                $runName: String,\n                $description: String,\n                $digest: String!,\n                $aliases: [ArtifactAliasInput!],\n                $metadata: JSONString,\n                _CREATE_ARTIFACT_ADDITIONAL_TYPE_\n            ) {\n                createArtifact(input: {\n                    artifactTypeName: $artifactTypeName,\n                    artifactCollectionNames: $artifactCollectionNames,\n                    entityName: $entityName,\n                    projectName: $projectName,\n                    runName: $runName,\n                    description: $description,\n                    digest: $digest,\n                    digestAlgorithm: MANIFEST_MD5,\n                    aliases: $aliases,\n                    metadata: $metadata,\n                    _CREATE_ARTIFACT_ADDITIONAL_VALUE_\n                }) {\n                    artifact {\n                        id\n                        state\n                        artifactSequence {\n                            id\n                            latestArtifact {\n                                id\n                                versionIndex\n                            }\n                        }\n                    }\n                }\n            }\n        '
    return query_template.replace('_CREATE_ARTIFACT_ADDITIONAL_TYPE_', types).replace('_CREATE_ARTIFACT_ADDITIONAL_VALUE_', values)