import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
def CalcLigRMSD(lig1, lig2, rename_lig2=True, output_filename='tmp.pdb'):
    """
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
    """
    lig1 = Chem.RemoveHs(lig1)
    lig2 = Chem.RemoveHs(lig2)
    coordinates_lig2 = lig2.GetConformer().GetPositions()
    coordinates_lig1 = lig1.GetConformer().GetPositions()
    res = rdFMCS.FindMCS([lig1, lig2])
    ref_mol = Chem.MolFromSmarts(res.smartsString)
    mas1 = list(lig1.GetSubstructMatch(ref_mol))
    mas2_list = lig2.GetSubstructMatches(ref_mol, uniquify=False)
    coordinates_lig1 = coordinates_lig1[mas1]
    list_rmsd = []
    for match1 in mas2_list:
        coordinates_lig2_tmp = coordinates_lig2[list(match1)]
        diff = coordinates_lig2_tmp - coordinates_lig1
        list_rmsd.append(np.sqrt((diff * diff).sum() / len(coordinates_lig2_tmp)))
    lig_rmsd = min(list_rmsd)
    if rename_lig2:
        mas2 = mas2_list[np.argmin(list_rmsd)]
        correspondence_key2_item1 = dict(zip(mas2, mas1))
        atom_names_lig1 = [atom1.GetPDBResidueInfo().GetName() for atom1 in lig1.GetAtoms()]
        lig1_ResName = lig1.GetAtoms()[0].GetPDBResidueInfo().GetResidueName()
        for i, atom1 in enumerate(lig2.GetAtoms()):
            atom1.GetPDBResidueInfo().SetResidueName(lig1_ResName)
            if i in correspondence_key2_item1.keys():
                atom1.GetPDBResidueInfo().SetName(atom_names_lig1[correspondence_key2_item1[i]])
        Chem.MolToPDBFile(lig2, output_filename)
    return lig_rmsd