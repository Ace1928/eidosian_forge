from aiokafka.metrics.measurable_stat import AbstractMeasurableStat
from aiokafka.metrics.stats.sampled_stat import AbstractSampledStat
class SampledTotal(AbstractSampledStat):

    def __init__(self, initial_value=None):
        if initial_value is not None:
            raise ValueError('initial_value cannot be set on SampledTotal')
        super(SampledTotal, self).__init__(0.0)

    def update(self, sample, config, value, time_ms):
        sample.value += value

    def combine(self, samples, config, now):
        return float(sum((sample.value for sample in samples)))