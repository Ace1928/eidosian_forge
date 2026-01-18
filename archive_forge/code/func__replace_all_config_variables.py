import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
def _replace_all_config_variables(self, collection: Union[Dict[str, Any], str], node_id: str, cluster_name: str, region: str) -> Union[str, Dict[str, Any]]:
    """
        Replace known config variable occurrences in the input collection.
        The input collection must be either a dict or list.
        Returns a tuple consisting of the output collection and the number of
        modified strings in the collection (which is not necessarily equal to
        the number of variables replaced).
        """
    for key in collection:
        if type(collection) is dict:
            value = collection.get(key)
            index_key = key
        elif type(collection) is list:
            value = key
            index_key = collection.index(key)
        else:
            raise ValueError(f"Can't replace CloudWatch config variables in unsupported collection type: {type(collection)}.Please check your CloudWatch JSON config files.")
        if type(value) is str:
            collection[index_key] = self._replace_config_variables(value, node_id, cluster_name, region)
        elif type(value) is dict or type(value) is list:
            collection[index_key] = self._replace_all_config_variables(value, node_id, cluster_name, region)
    return collection