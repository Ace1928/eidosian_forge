from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
from heat.engine import translation
class PrometheusAlarm(AodhBaseActionsMixin, alarm_base.BaseAlarm):
    """A resource that implements Aodh alarm of type prometheus.

    An alarm that evaluates threshold based on metric data fetched from
    Prometheus.
    """
    support_status = support.SupportStatus(version='22.0.0')
    PROPERTIES = COMPARISON_OPERATOR, QUERY, THRESHOLD = ('comparison_operator', 'query', 'threshold')
    properties_schema = {COMPARISON_OPERATOR: properties.Schema(properties.Schema.STRING, _('Operator used to compare specified statistic with threshold.'), constraints=[alarm_base.BaseAlarm.QF_OP_VALS], update_allowed=True), QUERY: properties.Schema(properties.Schema.STRING, _('The PromQL query string to fetch metrics data from Prometheus.'), required=True, update_allowed=True), THRESHOLD: properties.Schema(properties.Schema.NUMBER, _('Threshold to evaluate against.'), required=True, update_allowed=True)}
    properties_schema.update(alarm_base.common_properties_schema)
    alarm_type = 'prometheus'

    def get_alarm_props(self, props):
        kwargs = self.actions_to_urls(props)
        kwargs['type'] = self.alarm_type
        return self._reformat_properties(kwargs)

    def parse_live_resource_data(self, resource_properties, resource_data):
        record_reality = {}
        rule = self.alarm_type + '_rule'
        data = resource_data.get(rule).copy()
        data.update(resource_data)
        for key in self.properties_schema.keys():
            if key in alarm_base.INTERNAL_PROPERTIES:
                continue
            if self.properties_schema[key].update_allowed:
                record_reality.update({key: data.get(key)})
        return record_reality