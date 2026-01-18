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
class CliAlarmCreate(show.ShowOne):
    """Create an alarm"""
    create = True

    def get_parser(self, prog_name):
        parser = _add_name_to_parser(super(CliAlarmCreate, self).get_parser(prog_name), required=self.create)
        parser.add_argument('-t', '--type', metavar='<TYPE>', required=self.create, choices=ALARM_TYPES, help='Type of alarm, should be one of: %s.' % ', '.join(ALARM_TYPES))
        parser.add_argument('--project-id', metavar='<PROJECT_ID>', help='Project to associate with alarm (configurable by admin users only)')
        parser.add_argument('--user-id', metavar='<USER_ID>', help='User to associate with alarm (configurable by admin users only)')
        parser.add_argument('--description', metavar='<DESCRIPTION>', help='Free text description of the alarm')
        parser.add_argument('--state', metavar='<STATE>', choices=ALARM_STATES, help='State of the alarm, one of: ' + str(ALARM_STATES))
        parser.add_argument('--severity', metavar='<SEVERITY>', choices=ALARM_SEVERITY, help='Severity of the alarm, one of: ' + str(ALARM_SEVERITY))
        parser.add_argument('--enabled', type=strutils.bool_from_string, metavar='{True|False}', help='True if alarm evaluation is enabled')
        parser.add_argument('--alarm-action', dest='alarm_actions', metavar='<Webhook URL>', action='append', help='URL to invoke when state transitions to alarm. May be used multiple times')
        parser.add_argument('--ok-action', dest='ok_actions', metavar='<Webhook URL>', action='append', help='URL to invoke when state transitions to OK. May be used multiple times')
        parser.add_argument('--insufficient-data-action', dest='insufficient_data_actions', metavar='<Webhook URL>', action='append', help='URL to invoke when state transitions to insufficient data. May be used multiple times')
        parser.add_argument('--time-constraint', dest='time_constraints', metavar='<Time Constraint>', action='append', type=self.validate_time_constraint, help='Only evaluate the alarm if the time at evaluation is within this time constraint. Start point(s) of the constraint are specified with a cron expression, whereas its duration is given in seconds. Can be specified multiple times for multiple time constraints, format is: name=<CONSTRAINT_NAME>;start=<CRON>;duration=<SECONDS>;[description=<DESCRIPTION>;[timezone=<IANA Timezone>]]')
        parser.add_argument('--repeat-actions', dest='repeat_actions', metavar='{True|False}', type=strutils.bool_from_string, help='True if actions should be repeatedly notified while alarm remains in target state')
        common_group = parser.add_argument_group('common alarm rules')
        common_group.add_argument('--query', metavar='<QUERY>', dest='query', help='For alarms of type threshold or event: key[op]data_type::value; list. data_type is optional, but if supplied must be string, integer, float, or boolean. For alarms of type gnocchi_aggregation_by_resources_threshold: need to specify a complex query json string, like: {"and": [{"=": {"ended_at": null}}, ...]}. For alarms of type prometheus this should be valid PromQL query.')
        common_group.add_argument('--comparison-operator', metavar='<OPERATOR>', dest='comparison_operator', choices=ALARM_OPERATORS, help='Operator to compare with, one of: ' + str(ALARM_OPERATORS))
        common_group.add_argument('--evaluation-periods', type=int, metavar='<EVAL_PERIODS>', dest='evaluation_periods', help='Number of periods to evaluate over')
        common_group.add_argument('--threshold', type=float, metavar='<THRESHOLD>', dest='threshold', help='Threshold to evaluate against.')
        event_group = parser.add_argument_group('event alarm')
        event_group.add_argument('--event-type', metavar='<EVENT_TYPE>', dest='event_type', help='Event type to evaluate against')
        threshold_group = parser.add_argument_group('threshold alarm')
        threshold_group.add_argument('-m', '--meter-name', metavar='<METER NAME>', dest='meter_name', help='Meter to evaluate against')
        threshold_group.add_argument('--period', type=int, metavar='<PERIOD>', dest='period', help='Length of each period (seconds) to evaluate over.')
        threshold_group.add_argument('--statistic', metavar='<STATISTIC>', dest='statistic', choices=STATISTICS, help='Statistic to evaluate, one of: ' + str(STATISTICS))
        gnocchi_common_group = parser.add_argument_group('common gnocchi alarm rules')
        gnocchi_common_group.add_argument('--granularity', metavar='<GRANULARITY>', dest='granularity', help='The time range in seconds over which to query.')
        gnocchi_common_group.add_argument('--aggregation-method', metavar='<AGGR_METHOD>', dest='aggregation_method', help='The aggregation_method to compare to the threshold.')
        gnocchi_common_group.add_argument('--metric', '--metrics', metavar='<METRIC>', action='append', dest='metrics', help='The metric id or name depending of the alarm type')
        gnocchi_resource_threshold_group = parser.add_argument_group('gnocchi resource threshold alarm')
        gnocchi_resource_threshold_group.add_argument('--resource-type', metavar='<RESOURCE_TYPE>', dest='resource_type', help='The type of resource.')
        gnocchi_resource_threshold_group.add_argument('--resource-id', metavar='<RESOURCE_ID>', dest='resource_id', help='The id of a resource.')
        composite_group = parser.add_argument_group('composite alarm')
        composite_group.add_argument('--composite-rule', metavar='<COMPOSITE_RULE>', dest='composite_rule', type=jsonutils.loads, help='Composite threshold rule with JSON format, the form can be a nested dict which combine threshold/gnocchi rules by "and", "or". For example, the form is like: {"or":[RULE1, RULE2, {"and": [RULE3, RULE4]}]}, The RULEx can be basic threshold rules but must include a "type" field, like this: {"threshold": 0.8,"meter_name":"cpu_util","type":"threshold"}')
        loadbalancer_member_health_group = parser.add_argument_group('loadbalancer member health alarm')
        loadbalancer_member_health_group.add_argument('--stack-id', metavar='<STACK_NAME_OR_ID>', dest='stack_id', type=str, help='Name or ID of the root / top level Heat stack containing the loadbalancer pool and members. An update will be triggered on the root Stack if an unhealthy member is detected in the loadbalancer pool.')
        loadbalancer_member_health_group.add_argument('--pool-id', metavar='<LOADBALANCER_POOL_NAME_OR_ID>', dest='pool_id', type=str, help='Name or ID of the loadbalancer pool for which the health of each member will be evaluated.')
        loadbalancer_member_health_group.add_argument('--autoscaling-group-id', metavar='<AUTOSCALING_GROUP_NAME_OR_ID>', dest='autoscaling_group_id', type=str, help='ID of the Heat autoscaling group that contains the loadbalancer members. Unhealthy members will be marked as such before an update is triggered on the root stack.')
        self.parser = parser
        return parser

    def validate_time_constraint(self, values_to_convert):
        """Converts 'a=1;b=2' to {a:1,b:2}."""
        try:
            return dict(((item.strip(' "\'') for item in kv.split('=', 1)) for kv in values_to_convert.split(';')))
        except ValueError:
            msg = 'must be a list of key1=value1;key2=value2;... not %s' % values_to_convert
            raise argparse.ArgumentTypeError(msg)

    def _validate_args(self, parsed_args):
        if parsed_args.type == 'threshold' and (not (parsed_args.meter_name and parsed_args.threshold)):
            self.parser.error('Threshold alarm requires -m/--meter-name and --threshold parameters. Meter name can be found in Ceilometer')
        elif parsed_args.type == 'gnocchi_resources_threshold' and (not (parsed_args.metrics and parsed_args.threshold is not None and parsed_args.resource_id and parsed_args.resource_type and parsed_args.aggregation_method)):
            self.parser.error('gnocchi_resources_threshold requires --metric, --threshold, --resource-id, --resource-type and --aggregation-method')
        elif parsed_args.type == 'gnocchi_aggregation_by_metrics_threshold' and (not (parsed_args.metrics and parsed_args.threshold is not None and parsed_args.aggregation_method)):
            self.parser.error('gnocchi_aggregation_by_metrics_threshold requires --metric, --threshold and --aggregation-method')
        elif parsed_args.type == 'gnocchi_aggregation_by_resources_threshold' and (not (parsed_args.metrics and parsed_args.threshold is not None and parsed_args.query and parsed_args.resource_type and parsed_args.aggregation_method)):
            self.parser.error('gnocchi_aggregation_by_resources_threshold requires --metric, --threshold, --aggregation-method, --query and --resource-type')
        elif parsed_args.type == 'composite' and (not parsed_args.composite_rule):
            self.parser.error('Composite alarm requires --composite-rule parameter')
        elif parsed_args.type == 'loadbalancer_member_health' and (parsed_args.stack_id is None or parsed_args.pool_id is None or parsed_args.autoscaling_group_id is None):
            self.parser.error('Loadbalancer member health alarm requires--stack-id, --pool-id and--autoscaling-group-id')
        elif parsed_args.type == 'prometheus' and (not (parsed_args.query and parsed_args.threshold)):
            self.parser.error('Prometheus alarm requires --query and --threshold parameters.')

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

    def take_action(self, parsed_args):
        alarm = utils.get_client(self).alarm.create(alarm=self._alarm_from_args(parsed_args))
        return self.dict2columns(_format_alarm(alarm))