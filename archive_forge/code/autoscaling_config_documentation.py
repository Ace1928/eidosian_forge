import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
Fetch the RayCluster CR by querying the K8s API server.

        Retry on HTTPError for robustness, in particular to protect autoscaler
        initialization.
        