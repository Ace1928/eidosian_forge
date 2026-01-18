from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
from heat.engine import translation
class EventAlarm(AodhBaseActionsMixin, alarm_base.BaseAlarm):
    """A resource that implements event alarms.

    Allows users to define alarms which can be evaluated based on events
    passed from other OpenStack services. The events can be emitted when
    the resources from other OpenStack services have been updated, created
    or deleted, such as 'compute.instance.reboot.end',
    'scheduler.select_destinations.end'.
    """
    alarm_type = 'event'
    support_status = support.SupportStatus(version='8.0.0')
    PROPERTIES = EVENT_TYPE, QUERY = ('event_type', 'query')
    properties_schema = {EVENT_TYPE: properties.Schema(properties.Schema.STRING, _('Event type to evaluate against. If not specified will match all events.'), update_allowed=True, default='*'), QUERY: properties.Schema(properties.Schema.LIST, _('A list for filtering events. Query conditions used to filter specific events when evaluating the alarm.'), update_allowed=True, schema=properties.Schema(properties.Schema.MAP, schema={alarm_base.BaseAlarm.QF_FIELD: properties.Schema(properties.Schema.STRING, _('Name of attribute to compare.')), alarm_base.BaseAlarm.QF_TYPE: properties.Schema(properties.Schema.STRING, _('The type of the attribute.'), default='string', constraints=[alarm_base.BaseAlarm.QF_TYPE_VALS]), alarm_base.BaseAlarm.QF_OP: properties.Schema(properties.Schema.STRING, _('Comparison operator.'), constraints=[alarm_base.BaseAlarm.QF_OP_VALS]), alarm_base.BaseAlarm.QF_VALUE: properties.Schema(properties.Schema.STRING, _('String value with which to compare.'))}))}
    properties_schema.update(alarm_base.common_properties_schema)

    def get_alarm_props(self, props):
        """Apply all relevant compatibility xforms."""
        kwargs = self.actions_to_urls(props)
        kwargs['type'] = self.alarm_type
        rule = {}
        for prop in (self.EVENT_TYPE, self.QUERY):
            if prop in kwargs:
                del kwargs[prop]
        query = props.get(self.QUERY)
        if query:
            rule[self.QUERY] = query
        event_type = props.get(self.EVENT_TYPE)
        if event_type:
            rule[self.EVENT_TYPE] = event_type
        kwargs['event_rule'] = rule
        return kwargs