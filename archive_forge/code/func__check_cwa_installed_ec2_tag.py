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
def _check_cwa_installed_ec2_tag(self) -> List[str]:
    """
        Filtering all nodes to get nodes
        without Unified CloudWatch Agent installed
        """
    try:
        response = self.ec2_client.describe_instances(InstanceIds=[self.node_id])
        reservations = response['Reservations']
        message = 'More than 1 response received from describing current node'
        assert len(reservations) == 1, message
        instances = reservations[0]['Instances']
        assert len(instances) == 1, message
        tags = instances[0]['Tags']
        cwa_installed = str(False)
        for tag in tags:
            if tag['Key'] == CLOUDWATCH_AGENT_INSTALLED_TAG:
                logger.info('Unified CloudWatch Agent is installed on node {}'.format(self.node_id))
                cwa_installed = tag['Value']
        return cwa_installed
    except botocore.exceptions.ClientError as e:
        logger.warning('{} Error caught when getting Unified CloudWatch Agent status based on {} tag'.format(e.response['Error'], CLOUDWATCH_AGENT_INSTALLED_TAG))