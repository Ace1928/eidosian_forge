from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionAutoscalerAutoscalingpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'minNumReplicas': self.request.get('min_num_replicas'), u'maxNumReplicas': self.request.get('max_num_replicas'), u'coolDownPeriodSec': self.request.get('cool_down_period_sec'), u'mode': self.request.get('mode'), u'scaleInControl': RegionAutoscalerScaleincontrol(self.request.get('scale_in_control', {}), self.module).to_request(), u'cpuUtilization': RegionAutoscalerCpuutilization(self.request.get('cpu_utilization', {}), self.module).to_request(), u'customMetricUtilizations': RegionAutoscalerCustommetricutilizationsArray(self.request.get('custom_metric_utilizations', []), self.module).to_request(), u'loadBalancingUtilization': RegionAutoscalerLoadbalancingutilization(self.request.get('load_balancing_utilization', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'minNumReplicas': self.request.get(u'minNumReplicas'), u'maxNumReplicas': self.request.get(u'maxNumReplicas'), u'coolDownPeriodSec': self.request.get(u'coolDownPeriodSec'), u'mode': self.request.get(u'mode'), u'scaleInControl': RegionAutoscalerScaleincontrol(self.request.get(u'scaleInControl', {}), self.module).from_response(), u'cpuUtilization': RegionAutoscalerCpuutilization(self.request.get(u'cpuUtilization', {}), self.module).from_response(), u'customMetricUtilizations': RegionAutoscalerCustommetricutilizationsArray(self.request.get(u'customMetricUtilizations', []), self.module).from_response(), u'loadBalancingUtilization': RegionAutoscalerLoadbalancingutilization(self.request.get(u'loadBalancingUtilization', {}), self.module).from_response()})