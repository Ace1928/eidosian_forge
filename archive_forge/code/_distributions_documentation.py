import math
import cupy
from cupyx.scipy import special
Calculate the entropy of a distribution for given probability values.

    If only probabilities ``pk`` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If ``qk`` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize ``pk`` and ``qk`` if they don't sum to 1.

    Args:
        pk (ndarray): Defines the (discrete) distribution. ``pk[i]`` is the
            (possibly unnormalized) probability of event ``i``.
        qk (ndarray, optional): Sequence against which the relative entropy is
            computed. Should be in the same format as ``pk``.
        base (float, optional): The logarithmic base to use, defaults to ``e``
            (natural logarithm).
        axis (int, optional): The axis along which the entropy is calculated.
            Default is 0.

    Returns:
        S (cupy.ndarray): The calculated entropy.

    