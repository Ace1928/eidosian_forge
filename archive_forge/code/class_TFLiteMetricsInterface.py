import abc
class TFLiteMetricsInterface(metaclass=abc.ABCMeta):
    """Abstract class for TFLiteMetrics."""

    @abc.abstractmethod
    def increase_counter_debugger_creation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def increase_counter_interpreter_creation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def increase_counter_converter_attempt(self):
        raise NotImplementedError

    @abc.abstractmethod
    def increase_counter_converter_success(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_converter_param(self, name, value):
        raise NotImplementedError

    @abc.abstractmethod
    def set_converter_error(self, error_data):
        raise NotImplementedError

    @abc.abstractmethod
    def set_converter_latency(self, value):
        raise NotImplementedError