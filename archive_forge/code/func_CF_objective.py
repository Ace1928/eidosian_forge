import numpy as np
def CF_objective(L=None, A=None, T=None, kappa=0, rotation_method='orthogonal', return_gradient=True):
    """
    Objective function for the Crawford-Ferguson family for orthogonal
    and oblique rotation wich minimizes the following objective:

    .. math::
        \\phi(L) =\\frac{1-\\kappa}{4} (L\\circ L,(L\\circ L)N)
                  -\\frac{1}{4}(L\\circ L,M(L\\circ L)),

    where :math:`0\\leq\\kappa\\leq1`, :math:`L` is a :math:`p\\times k` matrix,
    :math:`N` is :math:`k\\times k` matrix with zeros on the diagonal and ones
    elsewhere,
    :math:`M` is :math:`p\\times p` matrix with zeros on the diagonal and ones
    elsewhere
    :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm and
    :math:`\\circ` is the element-wise product or Hadamard product.

    The gradient is given by

    .. math::
       d\\phi(L) = (1-\\kappa) L\\circ\\left[(L\\circ L)N\\right]
                   -\\kappa L\\circ \\left[M(L\\circ L)\\right].

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`
    satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    For orthogonal rotations the oblimin (and orthomax) family of rotations is
    equivalent to the Crawford-Ferguson family. To be more precise:

    * :math:`\\kappa=0` corresponds to quartimax,
    * :math:`\\kappa=\\frac{1}{p}` corresponds to variamx,
    * :math:`\\kappa=\\frac{k-1}{p+k-2}` corresponds to parsimax,
    * :math:`\\kappa=1` corresponds to factor parsimony.

    Parameters
    ----------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    rotation_method : str
        should be one of {orthogonal, oblique}
    return_gradient : bool (default True)
        toggles return of gradient
    """
    assert 0 <= kappa <= 1, 'Kappa should be between 0 and 1'
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    p, k = L.shape
    L2 = L ** 2
    X = None
    if not np.isclose(kappa, 1):
        N = np.ones((k, k)) - np.eye(k)
        X = (1 - kappa) * L2.dot(N)
    if not np.isclose(kappa, 0):
        M = np.ones((p, p)) - np.eye(p)
        if X is None:
            X = kappa * M.dot(L2)
        else:
            X += kappa * M.dot(L2)
    phi = np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = L * X
        return (phi, Gphi)
    else:
        return phi