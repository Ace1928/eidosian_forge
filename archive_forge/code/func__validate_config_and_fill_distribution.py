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
@staticmethod
def _validate_config_and_fill_distribution(config: dict) -> dict:
    config = deepcopy(config)
    config = dict(config)
    if 'parameters' not in config:
        return config
    for parameter_name in config['parameters']:
        parameter = config['parameters'][parameter_name]
        if 'min' in parameter and 'max' in parameter:
            if 'distribution' not in parameter:
                if isinstance(parameter['min'], int) and isinstance(parameter['max'], int):
                    parameter['distribution'] = 'int_uniform'
                elif isinstance(parameter['min'], float) and isinstance(parameter['max'], float):
                    parameter['distribution'] = 'uniform'
                else:
                    raise ValueError('Parameter %s is ambiguous, please specify bounds as both floats (for a float_uniform distribution) or ints (for an int_uniform distribution).' % parameter_name)
    return config