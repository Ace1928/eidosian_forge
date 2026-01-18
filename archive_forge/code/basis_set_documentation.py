from .basis_data import basis_sets, load_basisset
Generates default basis set parameters for a molecule.

    This function generates the default basis set parameters for a list of atomic symbols and
    computes the total number of basis functions for each atom.

    Args:
        name (str): name of the basis set
        symbols (list[str]): symbols of the atomic species in the molecule
        load_data (bool): flag to load data from the basis-set-exchange library

    Returns:
        tuple(list, tuple): the number of atomic basis functions and the basis set parameters for
        each atom in the molecule

    **Example**

    >>> n_basis, params = mol_basis_data('sto-3g', ['H', 'H'])
    >>> print(n_basis)
    [1, 1]
    >>> print(params)
    (((0, 0, 0), [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422]),
     ((0, 0, 0), [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422]))
    