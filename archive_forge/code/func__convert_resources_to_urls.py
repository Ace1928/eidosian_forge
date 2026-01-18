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
def _convert_resources_to_urls(self, configuration_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures that resources are in their full URL form.

        GCP expects machineType and acceleratorType to be a full URL (e.g.
        `zones/us-west1/machineTypes/n1-standard-2`) instead of just the
        type (`n1-standard-2`)

        Args:
            configuration_dict: Dict of options that will be passed to GCP
        Returns:
            Input dictionary, but with possibly expanding `machineType` and
                `acceleratorType`.
        """
    configuration_dict = deepcopy(configuration_dict)
    existing_machine_type = configuration_dict['machineType']
    if not re.search('.*/machineTypes/.*', existing_machine_type):
        configuration_dict['machineType'] = 'zones/{zone}/machineTypes/{machine_type}'.format(zone=self.availability_zone, machine_type=configuration_dict['machineType'])
    for accelerator in configuration_dict.get('guestAccelerators', []):
        gpu_type = accelerator['acceleratorType']
        if not re.search('.*/acceleratorTypes/.*', gpu_type):
            accelerator['acceleratorType'] = 'projects/{project}/zones/{zone}/acceleratorTypes/{accelerator}'.format(project=self.project_id, zone=self.availability_zone, accelerator=gpu_type)
    return configuration_dict