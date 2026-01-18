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
def _configure_subnet(config, compute):
    """Pick a reasonable subnet if not specified by the config."""
    config = copy.deepcopy(config)
    node_configs = [node_type['node_config'] for node_type in config['available_node_types'].values()]
    if all(('networkInterfaces' in node_config or 'networkConfig' in node_config for node_config in node_configs)):
        return config
    subnets = _list_subnets(config, compute)
    if not subnets:
        raise NotImplementedError('Should be able to create subnet.')
    default_subnet = subnets[0]
    default_interfaces = [{'subnetwork': default_subnet['selfLink'], 'accessConfigs': [{'name': 'External NAT', 'type': 'ONE_TO_ONE_NAT'}]}]
    for node_config in node_configs:
        if 'networkInterfaces' not in node_config:
            node_config['networkInterfaces'] = copy.deepcopy(default_interfaces)
        if 'networkConfig' not in node_config:
            node_config['networkConfig'] = copy.deepcopy(default_interfaces)[0]
            node_config['networkConfig'].pop('accessConfigs')
    return config