import collections
import inspect
import warnings
from functools import wraps
import tree
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import regularizers
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.distribution import distribution_lib
from keras.src.layers import input_spec
from keras.src.metrics.metric import Metric
from keras.src.ops.operation import Operation
from keras.src.utils import python_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils.shape_utils import map_shape_structure
class CallSpec:

    def __init__(self, signature, args, kwargs):
        if 'training' in kwargs and 'training' not in signature.parameters:
            kwargs.pop('training')
            bound_args = signature.bind(*args, **kwargs)
        else:
            bound_args = signature.bind(*args, **kwargs)
        self.user_arguments_dict = {k: v for k, v in bound_args.arguments.items()}
        bound_args.apply_defaults()
        arg_dict = {}
        arg_names = []
        tensor_arg_dict = {}
        tensor_args = []
        tensor_arg_names = []
        nested_tensor_arg_names = []
        for name, value in bound_args.arguments.items():
            arg_dict[name] = value
            arg_names.append(name)
            if is_backend_tensor_or_symbolic(value):
                tensor_args.append(value)
                tensor_arg_names.append(name)
                tensor_arg_dict[name] = value
            elif tree.is_nested(value):
                flat_values = tree.flatten(value)
                if all((is_backend_tensor_or_symbolic(x) for x in flat_values)):
                    tensor_args.append(value)
                    tensor_arg_names.append(name)
                    tensor_arg_dict[name] = value
                    nested_tensor_arg_names.append(name)
                elif any((is_backend_tensor_or_symbolic(x) for x in flat_values)):
                    raise ValueError(f'In a nested call() argument, you cannot mix tensors and non-tensors. Received invalid mixed argument: {name}={value}')
        self.arguments_dict = arg_dict
        self.argument_names = arg_names
        self.tensor_arguments_dict = tensor_arg_dict
        self.tensor_arguments_names = tensor_arg_names
        self.nested_tensor_argument_names = nested_tensor_arg_names
        self.first_arg = arg_dict[arg_names[0]]
        if all((backend.is_tensor(x) for x in self.tensor_arguments_dict.values())):
            self.eager = True
        else:
            self.eager = False