from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _GroupByReducerDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that groups its input and performs a reduction."""

    def __init__(self, input_dataset, key_func, reducer):
        """See `group_by_reducer()` for details."""
        self._input_dataset = input_dataset
        self._make_key_func(key_func, input_dataset)
        self._make_init_func(reducer.init_func)
        self._make_reduce_func(reducer.reduce_func, input_dataset)
        self._make_finalize_func(reducer.finalize_func)
        variant_tensor = ged_ops.experimental_group_by_reducer_dataset(self._input_dataset._variant_tensor, self._key_func.function.captured_inputs, self._init_func.function.captured_inputs, self._reduce_func.function.captured_inputs, self._finalize_func.function.captured_inputs, key_func=self._key_func.function, init_func=self._init_func.function, reduce_func=self._reduce_func.function, finalize_func=self._finalize_func.function, **self._flat_structure)
        super(_GroupByReducerDataset, self).__init__(input_dataset, variant_tensor)

    def _make_key_func(self, key_func, input_dataset):
        """Make wrapping defun for key_func."""
        self._key_func = structured_function.StructuredFunctionWrapper(key_func, self._transformation_name(), dataset=input_dataset)
        if not self._key_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int64)):
            raise ValueError(f'Invalid `key_func`. Expected `key_func` to return a scalar tf.int64 tensor, but instead `key_func` has output types={self._key_func.output_types} and shapes={self._key_func.output_shapes}.')

    def _make_init_func(self, init_func):
        """Make wrapping defun for init_func."""
        self._init_func = structured_function.StructuredFunctionWrapper(init_func, self._transformation_name(), input_structure=tensor_spec.TensorSpec([], dtypes.int64))

    def _make_reduce_func(self, reduce_func, input_dataset):
        """Make wrapping defun for reduce_func."""
        self._state_structure = self._init_func.output_structure
        state_types = self._init_func.output_types
        state_shapes = self._init_func.output_shapes
        state_classes = self._init_func.output_classes
        need_to_rerun = True
        while need_to_rerun:
            wrapped_func = structured_function.StructuredFunctionWrapper(reduce_func, self._transformation_name(), input_structure=(self._state_structure, input_dataset.element_spec), add_to_graph=False)
            for new_state_class, state_class in zip(nest.flatten(wrapped_func.output_classes), nest.flatten(state_classes)):
                if not issubclass(new_state_class, state_class):
                    raise TypeError(f'Invalid `reducer`. The output class of the `reducer.reduce_func` {wrapped_func.output_classes}, does not match the class of the reduce state {self._state_classes}.')
            for new_state_type, state_type in zip(nest.flatten(wrapped_func.output_types), nest.flatten(state_types)):
                if new_state_type != state_type:
                    raise TypeError(f'Invalid `reducer`. The element types for the new state {wrapped_func.output_types} do not match the element types of the old state {self._init_func.output_types}.')
            flat_state_shapes = nest.flatten(state_shapes)
            flat_new_state_shapes = nest.flatten(wrapped_func.output_shapes)
            weakened_state_shapes = [original.most_specific_compatible_shape(new) for original, new in zip(flat_state_shapes, flat_new_state_shapes)]
            need_to_rerun = False
            for original_shape, weakened_shape in zip(flat_state_shapes, weakened_state_shapes):
                if original_shape.ndims is not None and (weakened_shape.ndims is None or original_shape.as_list() != weakened_shape.as_list()):
                    need_to_rerun = True
                    break
            if need_to_rerun:
                state_shapes = nest.pack_sequence_as(self._init_func.output_shapes, weakened_state_shapes)
                self._state_structure = structure.convert_legacy_structure(state_types, state_shapes, state_classes)
        self._reduce_func = wrapped_func
        self._reduce_func.function.add_to_graph(ops.get_default_graph())

    def _make_finalize_func(self, finalize_func):
        """Make wrapping defun for finalize_func."""
        self._finalize_func = structured_function.StructuredFunctionWrapper(finalize_func, self._transformation_name(), input_structure=self._state_structure)

    @property
    def element_spec(self):
        return self._finalize_func.output_structure

    def _functions(self):
        return [self._key_func, self._init_func, self._reduce_func, self._finalize_func]

    def _transformation_name(self):
        return 'tf.data.experimental.group_by_reducer()'