import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
class Numerical(hyperparameter.HyperParameter):
    """Super class for all numerical type hyperparameters."""

    def __init__(self, name, min_value, max_value, step=None, sampling='linear', default=None, **kwargs):
        super().__init__(name=name, default=default, **kwargs)
        self.max_value = max_value
        self.min_value = min_value
        self.step = step
        self.sampling = sampling
        self._check_sampling_arg()

    def _check_sampling_arg(self):
        if self.min_value > self.max_value:
            raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), min_value {str(self.min_value)} is greater than the max_value {str(self.max_value)}.")
        sampling_values = {'linear', 'log', 'reverse_log'}
        if self.sampling is None:
            self.sampling = 'linear'
        self.sampling = self.sampling.lower()
        if self.sampling not in sampling_values:
            raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), sampling must be one of {sampling_values}")
        if self.sampling in {'log', 'reverse_log'} and self.min_value <= 0:
            raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), sampling='{str(self.sampling)}' does not support negative values, found min_value: {str(self.min_value)}.")
        if self.sampling in {'log', 'reverse_log'} and self.step is not None and (self.step <= 1):
            raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), expected step > 1 with sampling='{str(self.sampling)}'. Received: step={str(self.step)}.")

    def _sample_numerical_value(self, prob, max_value=None):
        """Sample a value with the cumulative prob in the given range."""
        if max_value is None:
            max_value = self.max_value
        if self.sampling == 'linear':
            return prob * (max_value - self.min_value) + self.min_value
        elif self.sampling == 'log':
            return self.min_value * math.pow(max_value / self.min_value, prob)
        elif self.sampling == 'reverse_log':
            return max_value + self.min_value - self.min_value * math.pow(max_value / self.min_value, 1 - prob)

    def _numerical_to_prob(self, value, max_value=None):
        """Convert a numerical value to range [0.0, 1.0)."""
        if max_value is None:
            max_value = self.max_value
        if max_value == self.min_value:
            return 0.5
        if self.sampling == 'linear':
            return (value - self.min_value) / (max_value - self.min_value)
        if self.sampling == 'log':
            return math.log(value / self.min_value) / math.log(max_value / self.min_value)
        if self.sampling == 'reverse_log':
            return 1.0 - math.log((max_value + self.min_value - value) / self.min_value) / math.log(max_value / self.min_value)

    def _get_n_values(self):
        """Get the total number of possible values using step."""
        if self.sampling == 'linear':
            return int((self.max_value - self.min_value) // self.step + 1)
        return int(math.log(self.max_value / self.min_value, self.step) + 1e-08) + 1

    def _get_value_by_index(self, index):
        """Get the index-th value in the range given step."""
        if self.sampling == 'linear':
            return self.min_value + index * self.step
        if self.sampling == 'log':
            return self.min_value * math.pow(self.step, index)
        if self.sampling == 'reverse_log':
            return self.max_value + self.min_value - self.min_value * math.pow(self.step, index)

    def _sample_with_step(self, prob):
        """Sample a value with the cumulative prob in the given range.

        The range is divided evenly by `step`. So only sampling from a finite
        set of values. When calling the function, no need to use (max_value + 1)
        since the function takes care of the inclusion of max_value.
        """
        n_values = self._get_n_values()
        index = hp_utils.prob_to_index(prob, n_values)
        return self._get_value_by_index(index)

    @property
    def values(self):
        if self.step is None:
            return tuple({self.prob_to_value(i * 0.1 + 0.05) for i in range(10)})
        n_values = self._get_n_values()
        return (self._get_value_by_index(i) for i in range(n_values))

    def _to_prob_with_step(self, value):
        """Convert to cumulative prob with step specified.

        When calling the function, no need to use (max_value + 1) since the
        function takes care of the inclusion of max_value.
        """
        if self.sampling == 'linear':
            index = (value - self.min_value) // self.step
        if self.sampling == 'log':
            index = math.log(value / self.min_value, self.step)
        if self.sampling == 'reverse_log':
            index = math.log((self.max_value - value + self.min_value) / self.min_value, self.step)
        n_values = self._get_n_values()
        return hp_utils.index_to_prob(index, n_values)