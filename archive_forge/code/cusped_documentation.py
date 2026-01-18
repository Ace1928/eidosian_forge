from ...sage_helper import _within_sage, sage_method
from .adjust_torsion import *
from .compute_ptolemys import *
from .. import verifyHyperbolicity
from ..cuspCrossSection import ComplexCuspCrossSection
from ...snap import t3mlite as t3m

    Computes the verified complex volume (where the real part is the
    volume and the imaginary part is the Chern-Simons) for a given
    SnapPy.Manifold.

    Note that the result is correct only up to two torsion, i.e.,
    up to multiples of pi^2/2. The method raises an exception if the
    manifold is not oriented or has a filled cusp.

    If bits_prec is unspecified, the default precision of
    SnapPy.Manifold, respectively, SnapPy.ManifoldHP will be used.
    