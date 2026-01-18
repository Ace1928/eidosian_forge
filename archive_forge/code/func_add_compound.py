import threading
import time
from aiokafka.errors import QuotaViolationError
from aiokafka.metrics.kafka_metric import KafkaMetric
def add_compound(self, compound_stat, config=None):
    """
        Register a compound statistic with this sensor which
        yields multiple measurable quantities (like a histogram)

        Arguments:
            stat (AbstractCompoundStat): The stat to register
            config (MetricConfig): The configuration for this stat.
                If None then the stat will use the default configuration
                for this sensor.
        """
    if not compound_stat:
        raise ValueError('compound stat must be non-empty')
    self._stats.append(compound_stat)
    for named_measurable in compound_stat.stats():
        metric = KafkaMetric(named_measurable.name, named_measurable.stat, config or self._config)
        self._registry.register_metric(metric)
        self._metrics.append(metric)