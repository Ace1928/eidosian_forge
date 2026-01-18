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
def _create_project_ssh_key_pair(project, public_key, ssh_user, compute):
    """Inserts an ssh-key into project commonInstanceMetadata"""
    key_parts = public_key.split(' ')
    assert len(key_parts) == 2, key_parts
    assert key_parts[0] == 'ssh-rsa', key_parts
    new_ssh_meta = '{ssh_user}:ssh-rsa {key_value} {ssh_user}'.format(ssh_user=ssh_user, key_value=key_parts[1])
    common_instance_metadata = project['commonInstanceMetadata']
    items = common_instance_metadata.get('items', [])
    ssh_keys_i = next((i for i, item in enumerate(items) if item['key'] == 'ssh-keys'), None)
    if ssh_keys_i is None:
        items.append({'key': 'ssh-keys', 'value': new_ssh_meta})
    else:
        ssh_keys = items[ssh_keys_i]
        ssh_keys['value'] += '\n' + new_ssh_meta
        items[ssh_keys_i] = ssh_keys
    common_instance_metadata['items'] = items
    operation = compute.projects().setCommonInstanceMetadata(project=project['name'], body=common_instance_metadata).execute()
    response = wait_for_compute_global_operation(project['name'], operation, compute)
    return response