import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_aws(provider_config):
    try:
        import boto3
    except ImportError as e:
        raise ImportError('The Ray AWS VM launcher requires the AWS SDK for Python (Boto3) to be installed. You can install it with `pip install boto3`.') from e
    from ray.autoscaler._private.aws.node_provider import AWSNodeProvider
    return AWSNodeProvider