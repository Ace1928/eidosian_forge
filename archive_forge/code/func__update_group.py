import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.group import ProcessType
from boto.ec2.autoscale.activity import Activity
from boto.ec2.autoscale.policy import AdjustmentType
from boto.ec2.autoscale.policy import MetricCollectionTypes
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.policy import TerminationPolicies
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.scheduled import ScheduledUpdateGroupAction
from boto.ec2.autoscale.tag import Tag
from boto.ec2.autoscale.limits import AccountLimits
from boto.compat import six
def _update_group(self, op, as_group):
    params = {'AutoScalingGroupName': as_group.name, 'LaunchConfigurationName': as_group.launch_config_name, 'MinSize': as_group.min_size, 'MaxSize': as_group.max_size}
    zones = as_group.availability_zones
    self.build_list_params(params, zones, 'AvailabilityZones')
    if as_group.desired_capacity is not None:
        params['DesiredCapacity'] = as_group.desired_capacity
    if as_group.vpc_zone_identifier:
        params['VPCZoneIdentifier'] = as_group.vpc_zone_identifier
    if as_group.health_check_period:
        params['HealthCheckGracePeriod'] = as_group.health_check_period
    if as_group.health_check_type:
        params['HealthCheckType'] = as_group.health_check_type
    if as_group.default_cooldown:
        params['DefaultCooldown'] = as_group.default_cooldown
    if as_group.placement_group:
        params['PlacementGroup'] = as_group.placement_group
    if as_group.instance_id:
        params['InstanceId'] = as_group.instance_id
    if as_group.termination_policies:
        self.build_list_params(params, as_group.termination_policies, 'TerminationPolicies')
    if op.startswith('Create'):
        if as_group.load_balancers:
            self.build_list_params(params, as_group.load_balancers, 'LoadBalancerNames')
        if as_group.tags:
            for i, tag in enumerate(as_group.tags):
                tag.build_params(params, i + 1)
    return self.get_object(op, params, Request)