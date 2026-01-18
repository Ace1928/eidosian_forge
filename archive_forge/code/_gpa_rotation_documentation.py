import numpy as np

    Subroutine for the value of vgQ using orthogonal rotation towards a partial
    target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2,

    where :math:`\circ` is the element-wise product or Hadamard product and
    :math:`W` is a matrix whose entries can only be one or zero. Either
    :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    Parameters
    ----------
    H : numpy matrix
        target matrix
    W : numpy matrix (default matrix with equal weight one for all entries)
        matrix with weights, entries can either be one or zero
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    