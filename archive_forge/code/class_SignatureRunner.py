import ctypes
import enum
import os
import platform
import sys
import numpy as np
class SignatureRunner:
    """SignatureRunner class for running TFLite models using SignatureDef.

  This class should be instantiated through TFLite Interpreter only using
  get_signature_runner method on Interpreter.
  Example,
  signature = interpreter.get_signature_runner("my_signature")
  result = signature(input_1=my_input_1, input_2=my_input_2)
  print(result["my_output"])
  print(result["my_second_output"])
  All names used are this specific SignatureDef names.

  Notes:
    No other function on this object or on the interpreter provided should be
    called while this object call has not finished.
  """

    def __init__(self, interpreter=None, signature_key=None):
        """Constructor.

    Args:
      interpreter: Interpreter object that is already initialized with the
        requested model.
      signature_key: SignatureDef key to be used.
    """
        if not interpreter:
            raise ValueError('None interpreter provided.')
        if not signature_key:
            raise ValueError('None signature_key provided.')
        self._interpreter = interpreter
        self._interpreter_wrapper = interpreter._interpreter
        self._signature_key = signature_key
        signature_defs = interpreter._get_full_signature_list()
        if signature_key not in signature_defs:
            raise ValueError('Invalid signature_key provided.')
        self._signature_def = signature_defs[signature_key]
        self._outputs = self._signature_def['outputs'].items()
        self._inputs = self._signature_def['inputs']
        self._subgraph_index = self._interpreter_wrapper.GetSubgraphIndexFromSignature(self._signature_key)

    def __call__(self, **kwargs):
        """Runs the SignatureDef given the provided inputs in arguments.

    Args:
      **kwargs: key,value for inputs to the model. Key is the SignatureDef input
        name. Value is numpy array with the value.

    Returns:
      dictionary of the results from the model invoke.
      Key in the dictionary is SignatureDef output name.
      Value is the result Tensor.
    """
        if len(kwargs) != len(self._inputs):
            raise ValueError('Invalid number of inputs provided for running a SignatureDef, expected %s vs provided %s' % (len(self._inputs), len(kwargs)))
        for input_name, value in kwargs.items():
            if input_name not in self._inputs:
                raise ValueError('Invalid Input name (%s) for SignatureDef' % input_name)
            self._interpreter_wrapper.ResizeInputTensor(self._inputs[input_name], np.array(value.shape, dtype=np.int32), False, self._subgraph_index)
        self._interpreter_wrapper.AllocateTensors(self._subgraph_index)
        for input_name, value in kwargs.items():
            self._interpreter_wrapper.SetTensor(self._inputs[input_name], value, self._subgraph_index)
        self._interpreter_wrapper.Invoke(self._subgraph_index)
        result = {}
        for output_name, output_index in self._outputs:
            result[output_name] = self._interpreter_wrapper.GetTensor(output_index, self._subgraph_index)
        return result

    def get_input_details(self):
        """Gets input tensor details.

    Returns:
      A dictionary from input name to tensor details where each item is a
      dictionary with details about an input tensor. Each dictionary contains
      the following fields that describe the tensor:

      + `name`: The tensor name.
      + `index`: The tensor index in the interpreter.
      + `shape`: The shape of the tensor.
      + `shape_signature`: Same as `shape` for models with known/fixed shapes.
        If any dimension sizes are unknown, they are indicated with `-1`.
      + `dtype`: The numpy data type (such as `np.int32` or `np.uint8`).
      + `quantization`: Deprecated, use `quantization_parameters`. This field
        only works for per-tensor quantization, whereas
        `quantization_parameters` works in all cases.
      + `quantization_parameters`: A dictionary of parameters used to quantize
        the tensor:
        ~ `scales`: List of scales (one if per-tensor quantization).
        ~ `zero_points`: List of zero_points (one if per-tensor quantization).
        ~ `quantized_dimension`: Specifies the dimension of per-axis
        quantization, in the case of multiple scales/zero_points.
      + `sparsity_parameters`: A dictionary of parameters used to encode a
        sparse tensor. This is empty if the tensor is dense.
    """
        result = {}
        for input_name, tensor_index in self._inputs.items():
            result[input_name] = self._interpreter._get_tensor_details(tensor_index, self._subgraph_index)
        return result

    def get_output_details(self):
        """Gets output tensor details.

    Returns:
      A dictionary from input name to tensor details where each item is a
      dictionary with details about an output tensor. The dictionary contains
      the same fields as described for `get_input_details()`.
    """
        result = {}
        for output_name, tensor_index in self._outputs:
            result[output_name] = self._interpreter._get_tensor_details(tensor_index, self._subgraph_index)
        return result