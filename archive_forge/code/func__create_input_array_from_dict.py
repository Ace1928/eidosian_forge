import numpy as np
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.framework import dtypes
from tensorflow.python.util.lazy_loader import LazyLoader
def _create_input_array_from_dict(self, signature_key, inputs):
    input_array = []
    signature_runner = self._interpreter.get_signature_runner(signature_key)
    input_details = sorted(signature_runner.get_input_details().items(), key=lambda item: item[1]['index'])
    for input_name, _ in input_details:
        input_array.append(inputs[input_name])
    return input_array