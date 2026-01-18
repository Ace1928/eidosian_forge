from heat.common.i18n import _
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine import support
class AodhGnocchiAggregationByMetricsAlarm(AodhGnocchiResourcesAlarm):
    """A resource that implements alarm with specified metrics.

    A resource that implements alarm which allows to use specified by user
    metrics in metrics list.
    """
    support_status = support.SupportStatus(version='2015.1')
    PROPERTIES = METRICS, = ('metrics',)
    PROPERTIES += COMMON_GNOCCHI_PROPERTIES
    properties_schema = {METRICS: properties.Schema(properties.Schema.LIST, _('A list of metric ids.'), required=True, update_allowed=True)}
    properties_schema.update(common_gnocchi_properties_schema)
    properties_schema.update(alarm_base.common_properties_schema)
    alarm_type = 'gnocchi_aggregation_by_metrics_threshold'