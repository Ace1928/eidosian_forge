import logging
import numpy as np
from mxnet.base import string_types
from mxnet import symbol
from ._export_onnx import MXNetGraph
from ._export_helper import load_module
Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage -
    https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#operator-support-matrix

    Parameters
    ----------
    sym : str or symbol object
        Path to the json file or Symbol object
    params : str or dict or list of dict
        str - Path to the params file
        dict - params dictionary (Including both arg_params and aux_params)
        list - list of length 2 that contains arg_params and aux_params
    in_shapes : List of tuple
        Input shape of the model e.g [(1,3,224,224)]
    in_types : data type or list of data types
        Input data type e.g. np.float32, or [np.float32, np.int32]
    onnx_file_path : str
        Path where to save the generated onnx file
    verbose : Boolean
        If True will print logs of the model conversion
    dynamic: Boolean
        If True will allow for dynamic input shapes to the model
    dynamic_input_shapes: list of tuple
        Specifies the dynamic input_shapes. If None then all dimensions are set to None
    run_shape_inference : Boolean
        If True will run shape inference on the model
    input_type : data type or list of data types
        This is the old name of in_types. We keep this parameter name for backward compatibility
    input_shape : List of tuple
        This is the old name of in_shapes. We keep this parameter name for backward compatibility

    Returns
    -------
    onnx_file_path : str
        Onnx file path

    Notes
    -----
    This method is available when you ``import mxnet.onnx``

    