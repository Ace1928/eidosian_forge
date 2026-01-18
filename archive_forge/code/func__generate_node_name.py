from API responses.
import abc
import logging
import re
import time
from collections import UserDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
def _generate_node_name(labels: dict, node_suffix: str) -> str:
    """Generate node name from labels and suffix.

    This is required so that the correct resource can be selected
    when the only information autoscaler has is the name of the node.

    The suffix is expected to be one of 'compute' or 'tpu'
    (as in ``GCPNodeType``).
    """
    name_label = labels[TAG_RAY_NODE_NAME]
    assert len(name_label) <= INSTANCE_NAME_MAX_LEN - INSTANCE_NAME_UUID_LEN - 1, (name_label, len(name_label))
    return f'{name_label}-{uuid4().hex[:INSTANCE_NAME_UUID_LEN]}-{node_suffix}'