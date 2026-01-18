from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateLayer7DdosDefenseThresholdConfig(client, args, support_granularity_config):
    """Returns a SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfig message."""
    messages = client.messages
    threshold_config = messages.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfig()
    threshold_config.name = args.threshold_config_name
    if args.IsSpecified('auto_deploy_load_threshold'):
        threshold_config.autoDeployLoadThreshold = args.auto_deploy_load_threshold
    if args.IsSpecified('auto_deploy_confidence_threshold'):
        threshold_config.autoDeployConfidenceThreshold = args.auto_deploy_confidence_threshold
    if args.IsSpecified('auto_deploy_impacted_baseline_threshold'):
        threshold_config.autoDeployImpactedBaselineThreshold = args.auto_deploy_impacted_baseline_threshold
    if args.IsSpecified('auto_deploy_expiration_sec'):
        threshold_config.autoDeployExpirationSec = args.auto_deploy_expiration_sec
    if support_granularity_config:
        if args.IsSpecified('detection_load_threshold'):
            threshold_config.detectionLoadThreshold = args.detection_load_threshold
        if args.IsSpecified('detection_absolute_qps'):
            threshold_config.detectionAbsoluteQps = args.detection_absolute_qps
        if args.IsSpecified('detection_relative_to_baseline_qps'):
            threshold_config.detectionRelativeToBaselineQps = args.detection_relative_to_baseline_qps
        if args.IsSpecified('traffic_granularity_configs'):
            traffic_granularity_configs = []
            for arg_traffic_granularity_config in args.traffic_granularity_configs:
                traffic_granularity_config = messages.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfigTrafficGranularityConfig()
                if 'type' in arg_traffic_granularity_config:
                    traffic_granularity_config.type = messages.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfigTrafficGranularityConfig.TypeValueValuesEnum(arg_traffic_granularity_config['type'])
                if 'value' in arg_traffic_granularity_config:
                    traffic_granularity_config.value = arg_traffic_granularity_config['value']
                if 'enableEachUniqueValue' in arg_traffic_granularity_config:
                    traffic_granularity_config.enableEachUniqueValue = arg_traffic_granularity_config['enableEachUniqueValue']
                traffic_granularity_configs.append(traffic_granularity_config)
            threshold_config.trafficGranularityConfigs = traffic_granularity_configs
    return threshold_config