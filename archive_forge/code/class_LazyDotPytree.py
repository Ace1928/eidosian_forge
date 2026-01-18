from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.tree_util import register_pytree_node_class
import pennylane as qml
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
@register_pytree_node_class
@dataclass
class LazyDotPytree:
    """Jax pytree representing a lazy dot operation."""
    coeffs: Tuple[complex, ...]
    mats: Tuple[Union[jnp.ndarray, sparse.BCSR], ...]

    @jax.jit
    def __matmul__(self, other):
        return sum((c * (m @ other) for c, m in zip(self.coeffs, self.mats)))

    def __mul__(self, other):
        if jnp.array(other).ndim == 0:
            return LazyDotPytree(tuple((other * c for c in self.coeffs)), self.mats)
        return NotImplemented
    __rmul__ = __mul__

    def tree_flatten(self):
        """Function used by ``jax`` to flatten the JaxLazyDot operation.

        Returns:
            tuple: tuple containing children and the auxiliary data of the class
        """
        children = (self.coeffs, self.mats)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Function used by ``jax`` to unflatten the ``JaxLazyDot`` pytree.

        Args:
            aux_data (None): empty argument
            children (tuple): tuple containing the coefficients and the matrices of the operation

        Returns:
            JaxLazyDot: JaxLazyDot instance
        """
        return cls(*children)