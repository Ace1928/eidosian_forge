import copy
import json
import logging
import os
import re
import time
from functools import partial, reduce
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from googleapiclient import discovery, errors
from ray._private.accelerators import TPUAcceleratorManager, tpu
from ray.autoscaler._private.gcp.node import MAX_POLLS, POLL_INTERVAL, GCPNodeType
from ray.autoscaler._private.util import check_legacy_fields
def _configure_project(config, crm):
    """Setup a Google Cloud Platform Project.

    Google Compute Platform organizes all the resources, such as storage
    buckets, users, and instances under projects. This is different from
    aws ec2 where everything is global.
    """
    config = copy.deepcopy(config)
    project_id = config['provider'].get('project_id')
    assert config['provider']['project_id'] is not None, "'project_id' must be set in the 'provider' section of the autoscaler config. Notice that the project id must be globally unique."
    project = _get_project(project_id, crm)
    if project is None:
        _create_project(project_id, crm)
        project = _get_project(project_id, crm)
    assert project is not None, 'Failed to create project'
    assert project['lifecycleState'] == 'ACTIVE', 'Project status needs to be ACTIVE, got {}'.format(project['lifecycleState'])
    config['provider']['project_id'] = project['projectId']
    return config