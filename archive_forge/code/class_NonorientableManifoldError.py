from .cuspCrossSection import ComplexCuspCrossSection
from .shapes import compute_hyperbolic_shapes
class NonorientableManifoldError(RuntimeError):
    """
    Exception raised when trying to compute cusp shapes for a non-orientable
    manifold.
    """

    def __init__(self, manifold):
        self.manifold = manifold

    def __str__(self):
        return 'Cannot compute cusp shapes for non-orientable manifold %s' % self.manifold