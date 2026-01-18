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
def _get_num_tpu_chips(node: dict) -> int:
    chips = 0
    if 'acceleratorType' in node:
        accelerator_type = node['acceleratorType']
        cores = int(accelerator_type.split('-')[1])
        chips = cores / tpu.TPU_CORES_PER_CHIP
    if 'acceleratorConfig' in node:
        topology = node['acceleratorConfig']['topology']
        chips = 1
        for dim in topology.split('x'):
            chips *= int(dim)
    return chips