from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
class _QidShapeSet:
    """A potentially infinite set of possible qid shapes."""

    def __init__(self, *, explicit_qid_shapes: Optional[Set[Tuple[int, ...]]]=None, unfactorized_total_dimension: Optional[int]=None, min_qudit_dimensions: Optional[Tuple[int, ...]]=None) -> None:
        """Create a qid shape set.

        The set of qid shapes is represented as the union of a set of shapes
        specified explicitly in `explicit_qid_shapes`, a set of shapes
        specified implicitly in `unfactorized_total_dimension`, and a set of
        shapes specified implicitly in `min_qudit_dimensions`.

        Args:
            explicit_qid_shapes: An explicit set of qid shapes.
            unfactorized_total_dimension: A number representing the dimension
                of the Hilbert space. The associated qid shapes are those compatible
                with this dimension, i.e., those for which the product of the
                individual qudit dimensions is equal to the Hilbert space
                dimension.
            min_qudit_dimensions: A tuple of integers (n_1, ..., n_k).
                The associated qid shapes are
                {(m_1, ..., m_k) : m_i ≥ n_i for all i}.
        """
        self.explicit_qid_shapes = explicit_qid_shapes or set()
        self.unfactorized_total_dimension = unfactorized_total_dimension
        self.min_qudit_dimensions = min_qudit_dimensions

    def intersection_subset(self, other: '_QidShapeSet'):
        """Return a subset of the intersection with other qid shape set."""
        explicit_qid_shapes = self.explicit_qid_shapes & other.explicit_qid_shapes
        unfactorized_total_dimension = None
        min_qudit_dimensions = None
        if self.explicit_qid_shapes and other.unfactorized_total_dimension is not None:
            explicit_qid_shapes |= _intersection_explicit_with_unfactorized_qid_shapes(self.explicit_qid_shapes, other.unfactorized_total_dimension)
        if self.explicit_qid_shapes and other.min_qudit_dimensions:
            explicit_qid_shapes |= _intersection_explicit_with_min_qudit_dims_qid_shapes(self.explicit_qid_shapes, other.min_qudit_dimensions)
        if self.unfactorized_total_dimension is not None and other.explicit_qid_shapes:
            explicit_qid_shapes |= _intersection_explicit_with_unfactorized_qid_shapes(other.explicit_qid_shapes, self.unfactorized_total_dimension)
        if self.unfactorized_total_dimension == other.unfactorized_total_dimension:
            unfactorized_total_dimension = self.unfactorized_total_dimension
        if self.min_qudit_dimensions is not None and other.explicit_qid_shapes:
            explicit_qid_shapes |= _intersection_explicit_with_min_qudit_dims_qid_shapes(other.explicit_qid_shapes, self.min_qudit_dimensions)
        if self.min_qudit_dimensions is not None and other.min_qudit_dimensions is not None:
            min_qudit_dimensions = _intersection_min_qudit_dims_qid_shapes(self.min_qudit_dimensions, other.min_qudit_dimensions)
        return _QidShapeSet(explicit_qid_shapes=explicit_qid_shapes, unfactorized_total_dimension=unfactorized_total_dimension, min_qudit_dimensions=min_qudit_dimensions)

    def _raise_value_error_if_ambiguous(self) -> None:
        """Raise an error if the qid shape is ambiguous and cannot be inferred."""
        if self.min_qudit_dimensions is not None:
            raise ValueError(f'Qid shape is ambiguous: Could be any shape on {len(self.min_qudit_dimensions)} qudits with the corresponding qudit dimensions being at least {self.min_qudit_dimensions}.')
        if len(self.explicit_qid_shapes) > 1:
            raise ValueError(f'Qid shape is ambiguous: Could be any one of {self.explicit_qid_shapes}.')
        if self.explicit_qid_shapes and self.unfactorized_total_dimension is not None:
            explicit_shape = next(iter(self.explicit_qid_shapes))
            raise ValueError(f'Qid shape is ambiguous: Could be {explicit_shape} or any shape compatible with a Hilbert space dimension of {self.unfactorized_total_dimension}.')

    def infer_qid_shape(self) -> Optional[Tuple[int, ...]]:
        """Return a qid shape from this set, or None."""
        self._raise_value_error_if_ambiguous()
        if self.unfactorized_total_dimension is not None:
            return _infer_qid_shape_from_dimension(self.unfactorized_total_dimension)
        if len(self.explicit_qid_shapes) == 0:
            return None
        return self.explicit_qid_shapes.pop()