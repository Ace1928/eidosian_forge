import abc
class AnonMeasurable(AbstractMeasurable):

    def __init__(self, measure_fn):
        self._measure_fn = measure_fn

    def measure(self, config, now):
        return float(self._measure_fn(config, now))