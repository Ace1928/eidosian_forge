from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
class BaseAlarm(resource.Resource):
    """Base Alarm Manager."""
    default_client_name = 'aodh'
    entity = 'alarm'
    alarm_type = 'threshold'
    QUERY_FACTOR_FIELDS = QF_FIELD, QF_OP, QF_VALUE, QF_TYPE = ('field', 'op', 'value', 'type')
    QF_OP_VALS = constraints.AllowedValues(['le', 'ge', 'eq', 'lt', 'gt', 'ne'])
    QF_TYPE_VALS = constraints.AllowedValues(['integer', 'float', 'string', 'boolean', 'datetime'])

    def actions_to_urls(self, props):
        kwargs = dict(props)

        def get_urls(action_type, queue_type):
            for act in kwargs.get(action_type) or []:
                if act in self.stack:
                    yield self.stack[act].FnGetAtt('AlarmUrl')
                elif act:
                    yield act
            for queue in kwargs.pop(queue_type, []):
                query = {'queue_name': queue}
                yield ('trust+zaqar://?%s' % parse.urlencode(query))
        action_props = {arg_types[0]: list(get_urls(*arg_types)) for arg_types in ((ALARM_ACTIONS, ALARM_QUEUES), (OK_ACTIONS, OK_QUEUES), (INSUFFICIENT_DATA_ACTIONS, INSUFFICIENT_DATA_QUEUES))}
        kwargs.update(action_props)
        return kwargs

    def _reformat_properties(self, props):
        rule = {}
        for name in self.PROPERTIES:
            if name in props:
                rule[name] = props.pop(name)
        if rule:
            props['%s_rule' % self.alarm_type] = rule
        return props

    def handle_suspend(self):
        if self.resource_id is not None:
            alarm_update = {'enabled': False}
            self.client().alarm.update(self.resource_id, alarm_update)

    def handle_resume(self):
        if self.resource_id is not None:
            alarm_update = {'enabled': True}
            self.client().alarm.update(self.resource_id, alarm_update)

    def handle_check(self):
        self.client().alarm.get(self.resource_id)