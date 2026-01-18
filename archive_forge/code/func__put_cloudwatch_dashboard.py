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
def _put_cloudwatch_dashboard(self) -> Dict[str, Any]:
    """put dashboard to cloudwatch console"""
    cloudwatch_config = self.provider_config['cloudwatch']
    dashboard_config = cloudwatch_config.get('dashboard', {})
    dashboard_name_cluster = dashboard_config.get('name', self.cluster_name)
    dashboard_name = self.cluster_name + '-' + dashboard_name_cluster
    widgets = self._replace_dashboard_config_vars(CloudwatchConfigType.DASHBOARD.value)
    response = self.cloudwatch_client.put_dashboard(DashboardName=dashboard_name, DashboardBody=json.dumps({'widgets': widgets}))
    issue_count = len(response.get('DashboardValidationMessages', []))
    if issue_count > 0:
        for issue in response.get('DashboardValidationMessages'):
            logging.error('Error in dashboard config: {} - {}'.format(issue['Message'], issue['DataPath']))
        raise Exception('Errors in dashboard configuration: {} issues raised'.format(issue_count))
    else:
        logger.info('Successfully put dashboard to CloudWatch console')
    return response