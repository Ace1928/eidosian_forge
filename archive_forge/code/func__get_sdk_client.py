import json
import os
import sys
import pprint
import time
from subprocess import list2cmdline
from typing import Optional, Tuple, Union, Dict, Any
import click
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
from ray._private.utils import get_or_create_event_loop
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args
from ray.job_submission import JobStatus, JobSubmissionClient
from ray.dashboard.modules.job.cli_utils import add_common_job_options
from ray.dashboard.modules.job.utils import redact_url_password
from ray.util.annotations import PublicAPI
from ray._private.utils import parse_resources_json, parse_metadata_json
def _get_sdk_client(address: Optional[str], create_cluster_if_needed: bool=False, headers: Optional[str]=None, verify: Union[bool, str]=True) -> JobSubmissionClient:
    client = JobSubmissionClient(address, create_cluster_if_needed, headers=_handle_headers(headers), verify=verify)
    client_address = client.get_address()
    cli_logger.labeled_value('Job submission server address', redact_url_password(client_address))
    return client