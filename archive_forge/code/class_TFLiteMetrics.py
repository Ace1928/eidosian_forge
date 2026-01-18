import os
from typing import Optional, Text
class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
    """TFLite metrics helper."""

    def __init__(self, model_hash: Optional[Text]=None, model_path: Optional[Text]=None) -> None:
        pass

    def increase_counter_debugger_creation(self):
        pass

    def increase_counter_interpreter_creation(self):
        pass

    def increase_counter_converter_attempt(self):
        pass

    def increase_counter_converter_success(self):
        pass

    def set_converter_param(self, name, value):
        pass

    def set_converter_error(self, error_data):
        pass

    def set_converter_latency(self, value):
        pass