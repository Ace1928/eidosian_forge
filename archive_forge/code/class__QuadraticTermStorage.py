from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
class _QuadraticTermStorage:
    """Data describing quadratic terms with non-zero coefficients."""

    def __init__(self) -> None:
        self._coefficients: Dict[_QuadraticKey, float] = {}
        self._adjacent_variables: Dict[int, Set[int]] = {}

    def __bool__(self) -> bool:
        """Returns true if and only if there are any quadratic terms with non-zero coefficients."""
        return bool(self._coefficients)

    def get_adjacent_variables(self, variable_id: int) -> Iterator[int]:
        """Yields the variables multiplying a variable in the stored quadratic terms.

        If variable_id is not in the model the function yields the empty set.

        Args:
          variable_id: Function yields the variables multiplying variable_id in the
            stored quadratic terms.

        Yields:
          The variables multiplying variable_id in the stored quadratic terms.
        """
        yield from self._adjacent_variables.get(variable_id, ())

    def keys(self) -> Iterator[_QuadraticKey]:
        """Yields the variable-pair keys associated to the stored quadratic terms."""
        yield from self._coefficients.keys()

    def coefficients(self) -> Iterator[model_storage.QuadraticEntry]:
        """Yields the stored quadratic terms as QuadraticEntry."""
        for key, coef in self._coefficients.items():
            yield model_storage.QuadraticEntry(id_key=key, coefficient=coef)

    def delete_variable(self, variable_id: int) -> None:
        """Updates the data structure to consider variable_id as deleted."""
        if variable_id not in self._adjacent_variables.keys():
            return
        for adjacent_variable_id in self._adjacent_variables[variable_id]:
            if variable_id != adjacent_variable_id:
                self._adjacent_variables[adjacent_variable_id].remove(variable_id)
            del self._coefficients[_QuadraticKey(variable_id, adjacent_variable_id)]
        self._adjacent_variables[variable_id].clear()

    def clear(self) -> None:
        """Clears the data structure."""
        self._coefficients.clear()
        self._adjacent_variables.clear()

    def set_coefficient(self, first_variable_id: int, second_variable_id: int, value: float) -> bool:
        """Sets the coefficient for the quadratic term associated to the product between two variables.

        The ordering of the input variables does not matter.

        Args:
          first_variable_id: The first variable in the product.
          second_variable_id: The second variable in the product.
          value: The value of the coefficient.

        Returns:
          True if the coefficient is updated, False otherwise.
        """
        key = _QuadraticKey(first_variable_id, second_variable_id)
        if value == self._coefficients.get(key, 0.0):
            return False
        if value == 0.0:
            del self._coefficients[key]
            self._adjacent_variables[first_variable_id].remove(second_variable_id)
            if first_variable_id != second_variable_id:
                self._adjacent_variables[second_variable_id].remove(first_variable_id)
        else:
            if first_variable_id not in self._adjacent_variables.keys():
                self._adjacent_variables[first_variable_id] = set()
            if second_variable_id not in self._adjacent_variables.keys():
                self._adjacent_variables[second_variable_id] = set()
            self._coefficients[key] = value
            self._adjacent_variables[first_variable_id].add(second_variable_id)
            self._adjacent_variables[second_variable_id].add(first_variable_id)
        return True

    def get_coefficient(self, first_variable_id: int, second_variable_id: int) -> float:
        """Gets the objective coefficient for the quadratic term associated to the product between two variables.

        The ordering of the input variables does not matter.

        Args:
          first_variable_id: The first variable in the product.
          second_variable_id: The second variable in the product.

        Returns:
          The value of the coefficient.
        """
        return self._coefficients.get(_QuadraticKey(first_variable_id, second_variable_id), 0.0)