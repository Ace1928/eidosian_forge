import os
import pathlib
import re
import sys
import time
import traceback
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import click
import watchfiles
import yaml
import ray
from ray import serve
from ray._private.utils import import_attr
from ray.autoscaler._private.cli_logger import cli_logger
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve._private import api as _private_api
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve.config import DeploymentMode, ProxyLocation, gRPCOptions
from ray.serve.deployment import Application, deployment_to_schema
from ray.serve.schema import (
class ServeDeploySchemaDumper(yaml.SafeDumper):
    """YAML dumper object with custom formatting for ServeDeploySchema.

    Reformat config to follow this spacing:
    ---------------------------------------

    host: 0.0.0.0

    port: 8000

    applications:

    - name: app1

      import_path: app1.path

      runtime_env: {}

      deployments:

      - name: deployment1
        ...

      - name: deployment2
        ...
    """

    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) <= 4:
            super().write_line_break()