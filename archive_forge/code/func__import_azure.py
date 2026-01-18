import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_azure(provider_config):
    from ray.autoscaler._private._azure.node_provider import AzureNodeProvider
    return AzureNodeProvider