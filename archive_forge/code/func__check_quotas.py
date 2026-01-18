import threading
import time
from aiokafka.errors import QuotaViolationError
from aiokafka.metrics.kafka_metric import KafkaMetric
def _check_quotas(self, time_ms):
    """
        Check if we have violated our quota for any metric that
        has a configured quota
        """
    for metric in self._metrics:
        if metric.config and metric.config.quota:
            value = metric.value(time_ms)
            if not metric.config.quota.is_acceptable(value):
                raise QuotaViolationError("'%s' violated quota. Actual: %d, Threshold: %d" % (metric.metric_name, value, metric.config.quota.bound))