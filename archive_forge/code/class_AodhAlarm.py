from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
from heat.engine import translation
class AodhAlarm(AodhBaseActionsMixin, alarm_base.BaseAlarm):
    """A resource that implements alarming service of Aodh.

    A resource that allows for the setting alarms based on threshold evaluation
    for a collection of samples. Also, you can define actions to take if state
    of watched resource will be satisfied specified conditions. For example, it
    can watch for the memory consumption and when it reaches 70% on a given
    instance if the instance has been up for more than 10 min, some action will
    be called.
    """
    support_status = support.SupportStatus(status=support.DEPRECATED, message=_('Theshold alarm relies on ceilometer-api and has been deprecated in aodh since Ocata. Use OS::Aodh::GnocchiAggregationByResourcesAlarm instead.'), version='10.0.0', previous_status=support.SupportStatus(version='2014.1'))
    PROPERTIES = COMPARISON_OPERATOR, EVALUATION_PERIODS, METER_NAME, PERIOD, STATISTIC, THRESHOLD, MATCHING_METADATA, QUERY = ('comparison_operator', 'evaluation_periods', 'meter_name', 'period', 'statistic', 'threshold', 'matching_metadata', 'query')
    properties_schema = {COMPARISON_OPERATOR: properties.Schema(properties.Schema.STRING, _('Operator used to compare specified statistic with threshold.'), constraints=[alarm_base.BaseAlarm.QF_OP_VALS], update_allowed=True), EVALUATION_PERIODS: properties.Schema(properties.Schema.INTEGER, _('Number of periods to evaluate over.'), update_allowed=True), METER_NAME: properties.Schema(properties.Schema.STRING, _('Meter name watched by the alarm.'), required=True), PERIOD: properties.Schema(properties.Schema.INTEGER, _('Period (seconds) to evaluate over.'), update_allowed=True), STATISTIC: properties.Schema(properties.Schema.STRING, _('Meter statistic to evaluate.'), constraints=[constraints.AllowedValues(['count', 'avg', 'sum', 'min', 'max'])], update_allowed=True), THRESHOLD: properties.Schema(properties.Schema.NUMBER, _('Threshold to evaluate against.'), required=True, update_allowed=True), MATCHING_METADATA: properties.Schema(properties.Schema.MAP, _('Meter should match this resource metadata (key=value) additionally to the meter_name.'), default={}, update_allowed=True), QUERY: properties.Schema(properties.Schema.LIST, _('A list of query factors, each comparing a Sample attribute with a value. Implicitly combined with matching_metadata, if any.'), update_allowed=True, support_status=support.SupportStatus(version='2015.1'), schema=properties.Schema(properties.Schema.MAP, schema={alarm_base.BaseAlarm.QF_FIELD: properties.Schema(properties.Schema.STRING, _('Name of attribute to compare. Names of the form metadata.user_metadata.X or metadata.metering.X are equivalent to what you can address through matching_metadata; the former for Nova meters, the latter for all others. To see the attributes of your Samples, use `ceilometer --debug sample-list`.')), alarm_base.BaseAlarm.QF_TYPE: properties.Schema(properties.Schema.STRING, _('The type of the attribute.'), default='string', constraints=[alarm_base.BaseAlarm.QF_TYPE_VALS], support_status=support.SupportStatus(version='8.0.0')), alarm_base.BaseAlarm.QF_OP: properties.Schema(properties.Schema.STRING, _('Comparison operator.'), constraints=[alarm_base.BaseAlarm.QF_OP_VALS]), alarm_base.BaseAlarm.QF_VALUE: properties.Schema(properties.Schema.STRING, _('String value with which to compare.'))}))}
    properties_schema.update(alarm_base.common_properties_schema)

    def get_alarm_props(self, props):
        """Apply all relevant compatibility xforms."""
        kwargs = self.actions_to_urls(props)
        kwargs['type'] = self.alarm_type
        if kwargs.get(self.METER_NAME) in alarm_base.NOVA_METERS:
            prefix = 'user_metadata.'
        else:
            prefix = 'metering.'
        rule = {}
        for field in ['period', 'evaluation_periods', 'threshold', 'statistic', 'comparison_operator', 'meter_name']:
            if field in kwargs:
                rule[field] = kwargs[field]
                del kwargs[field]
        mmd = props.get(self.MATCHING_METADATA) or {}
        query = props.get(self.QUERY) or []
        for m_k, m_v in mmd.items():
            key = 'metadata.%s' % prefix
            if m_k.startswith('metadata.'):
                m_k = m_k[len('metadata.'):]
            if m_k.startswith('metering.') or m_k.startswith('user_metadata.'):
                m_k = m_k.split('.', 1)[-1]
            key = '%s%s' % (key, m_k)
            query.append(dict(field=key, op='eq', value=str(m_v)))
        if self.MATCHING_METADATA in kwargs:
            del kwargs[self.MATCHING_METADATA]
        if self.QUERY in kwargs:
            del kwargs[self.QUERY]
        if query:
            rule['query'] = query
        kwargs['threshold_rule'] = rule
        return kwargs

    def parse_live_resource_data(self, resource_properties, resource_data):
        record_reality = {}
        threshold_data = resource_data.get('threshold_rule').copy()
        threshold_data.update(resource_data)
        props_upd_allowed = set(self.PROPERTIES + alarm_base.COMMON_PROPERTIES) - {self.METER_NAME, alarm_base.TIME_CONSTRAINTS} - set(alarm_base.INTERNAL_PROPERTIES)
        for key in props_upd_allowed:
            record_reality.update({key: threshold_data.get(key)})
        return record_reality

    def handle_check(self):
        self.client().alarm.get(self.resource_id)