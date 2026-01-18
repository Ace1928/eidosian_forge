import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
def _alarm_from_args(self, parsed_args):
    alarm = utils.dict_from_parsed_args(parsed_args, ['name', 'type', 'project_id', 'user_id', 'description', 'state', 'severity', 'enabled', 'alarm_actions', 'ok_actions', 'insufficient_data_actions', 'time_constraints', 'repeat_actions'])
    if parsed_args.type in ('threshold', 'event') and parsed_args.query:
        parsed_args.query = utils.cli_to_array(parsed_args.query)
    alarm['threshold_rule'] = utils.dict_from_parsed_args(parsed_args, ['meter_name', 'period', 'evaluation_periods', 'statistic', 'comparison_operator', 'threshold', 'query'])
    alarm['event_rule'] = utils.dict_from_parsed_args(parsed_args, ['event_type', 'query'])
    alarm['prometheus_rule'] = utils.dict_from_parsed_args(parsed_args, ['comparison_operator', 'threshold', 'query'])
    alarm['gnocchi_resources_threshold_rule'] = utils.dict_from_parsed_args(parsed_args, ['granularity', 'comparison_operator', 'threshold', 'aggregation_method', 'evaluation_periods', 'metric', 'resource_id', 'resource_type'])
    alarm['gnocchi_aggregation_by_metrics_threshold_rule'] = utils.dict_from_parsed_args(parsed_args, ['granularity', 'comparison_operator', 'threshold', 'aggregation_method', 'evaluation_periods', 'metrics'])
    alarm['gnocchi_aggregation_by_resources_threshold_rule'] = utils.dict_from_parsed_args(parsed_args, ['granularity', 'comparison_operator', 'threshold', 'aggregation_method', 'evaluation_periods', 'metric', 'query', 'resource_type'])
    alarm['loadbalancer_member_health_rule'] = utils.dict_from_parsed_args(parsed_args, ['stack_id', 'pool_id', 'autoscaling_group_id'])
    alarm['composite_rule'] = parsed_args.composite_rule
    if self.create:
        alarm['type'] = parsed_args.type
        self._validate_args(parsed_args)
    return alarm