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
def _get_subnet(config, subnet_id, compute):
    subnet = compute.subnetworks().get(project=config['provider']['project_id'], region=config['provider']['region'], subnetwork=subnet_id).execute()
    return subnet