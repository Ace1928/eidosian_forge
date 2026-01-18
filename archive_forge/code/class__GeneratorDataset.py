import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import script_ops
class _GeneratorDataset(dataset_ops.DatasetSource):
    """A `Dataset` that generates elements by invoking a function."""

    def __init__(self, init_args, init_func, next_func, finalize_func, output_signature, name=None):
        """Constructs a `_GeneratorDataset`.

    Args:
      init_args: A (nested) structure representing the arguments to `init_func`.
      init_func: A TensorFlow function that will be called on `init_args` each
        time a C++ iterator over this dataset is constructed. Returns a (nested)
        structure representing the "state" of the dataset.
      next_func: A TensorFlow function that will be called on the result of
        `init_func` to produce each element, and that raises `OutOfRangeError`
        to terminate iteration.
      finalize_func: A TensorFlow function that will be called on the result of
        `init_func` immediately before a C++ iterator over this dataset is
        destroyed. The return value is ignored.
      output_signature: A (nested) structure of `tf.TypeSpec` objects describing
        the output of `next_func`.
      name: Optional. A name for the tf.data transformation.
    """
        self._init_args = init_args
        self._init_structure = structure.type_spec_from_value(init_args)
        self._init_func = structured_function.StructuredFunctionWrapper(init_func, self._transformation_name(), input_structure=self._init_structure)
        self._next_func = structured_function.StructuredFunctionWrapper(next_func, self._transformation_name(), input_structure=self._init_func.output_structure)
        self._finalize_func = structured_function.StructuredFunctionWrapper(finalize_func, self._transformation_name(), input_structure=self._init_func.output_structure)
        self._output_signature = output_signature
        self._name = name
        variant_tensor = gen_dataset_ops.generator_dataset(structure.to_tensor_list(self._init_structure, self._init_args) + self._init_func.function.captured_inputs, self._next_func.function.captured_inputs, self._finalize_func.function.captured_inputs, init_func=self._init_func.function, next_func=self._next_func.function, finalize_func=self._finalize_func.function, **self._common_args)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._output_signature

    def _transformation_name(self):
        return 'Dataset.from_generator()'