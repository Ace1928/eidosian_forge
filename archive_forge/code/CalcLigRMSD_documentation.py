import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

    Calculate the Root-mean-square deviation (RMSD) between two prealigned ligands, 
    even when atom names between the two ligands are not matching.
    The symmetry of the molecules is taken into consideration (e.g. tri-methyl groups). 
    Moreover, if one ligand structure has missing atoms (e.g. undefined electron density in the crystal structure), 
    the RMSD is calculated for the maximum common substructure (MCS).

    Parameters
    ----------
    lig1 : RDKit molecule
    lig2 : RDKit molecule
    rename_lig2 : bool, optional
        True to rename the atoms of lig2 according to the atom names of lig1
    output_filename : str, optional
        If rename_lig2 is set to True, a PDB file with the renamed lig2 atoms is written as output.
        This may be useful to check that the RMSD has been "properly" calculated, 
        i.e. that the atoms have been properly matched for the calculation of the RMSD.
    
    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two input molecules
    