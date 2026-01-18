from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RDF, pi, matrix, block_matrix, vector
def _compute_errors_with_norm(hyperbolicStructure):
    global _two_pi
    errors = [angle - _two_pi for angle in hyperbolicStructure.angle_sums]
    return (errors, vector(errors).norm())