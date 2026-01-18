import itertools
Extracts basis set data from the Basis Set Exchange library.

    Args:
        basis (str): name of the basis set
        element (str): atomic symbol of the chemical element

    Returns:
        dict[str, list]: dictionary containing orbital names, and the exponents and contraction
        coefficients of a basis function

    **Example**

    >>> basis = '6-31g'
    >>> element = 'He'
    >>> basis = qml.qchem.load_basisset(basis, element)
    >>> basis
    {'orbitals': ['S', 'S'],
     'exponents': [[38.421634, 5.77803, 1.241774], [0.297964]],
     'coefficients': [[0.04013973935, 0.261246097, 0.7931846246], [1.0]]}
    