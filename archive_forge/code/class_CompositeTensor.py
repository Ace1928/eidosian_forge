import abc
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.CompositeTensor', v1=[])
class CompositeTensor(metaclass=abc.ABCMeta):
    """Abstract base class for Tensor-like objects that are composed from Tensors.

  Each `CompositeTensor` can be decomposed into a structured collection of
  component `tf.Tensor`s, and reconstructed from those components.

  The `tensorflow.python.util.nest` module has support for treating composite
  tensors as structure, which makes it easy to flatten and reconstruct
  composite tensors (or larger structures that contain composite tensors).
  E.g.:

  ```python
  ct = ...  # Create a composite tensor.
  flat_list_of_tensors = nest.flatten(ct, expand_composites=True)
  transformed_list_of_tensors = ...  # do something with the flat tensors.
  result = nest.pack_sequence_as(ct, transformed_list_of_tensors,
                                 expand_composites=True)
  ```
  """

    @abc.abstractproperty
    def _type_spec(self):
        """A `TypeSpec` describing the type of this value."""
        raise NotImplementedError(f'{type(self).__name__}._type_spec()')

    def _shape_invariant_to_type_spec(self, shape):
        """Returns a TypeSpec given a shape invariant (used by `tf.while_loop`).

    Args:
      shape: A `tf.TensorShape` object.  The shape invariant for this
        `CompositeTensor`, or `None` if a default shape invariant should be used
        (based on the value of this `CompositeTensor`).

    Returns:
      A nested structure whose values are `tf.TensorShape` objects, specifying
      the shape invariants for the tensors that comprise this `CompositeTensor`.
    """
        raise NotImplementedError(f'{type(self).__name__}._shape_invariant_to_type_spec')

    def _consumers(self):
        """Returns a list of `Operation`s that consume this `CompositeTensor`.

    Returns:
      A list of `Operation`s.

    Raises:
      RuntimeError: If this method is called while executing eagerly.
    """
        consumers = nest.flatten([component.consumers() for component in nest.flatten(self, expand_composites=True) if getattr(component, 'graph', None) is not None])
        return list(set(consumers))

    def __tf_tracing_type__(self, context):
        return self._type_spec.__tf_tracing_type__(context)

    def _convert_variables_to_tensors(self):
        """Converts ResourceVariable components to Tensors.

    Override this method to explicitly convert ResourceVariables embedded in the
    CompositeTensor to Tensors. By default, it returns the CompositeTensor
    unchanged.

    Returns:
      A CompositeTensor with all its ResourceVariable components converted to
      Tensors.
    """
        return self