import sys
import warnings
from collections import namedtuple
import numpy
from rdkit import DataStructs, ForceField, RDConfig, rdBase
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdFingerprintGenerator import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Geometry import rdGeometry
from rdkit.RDLogger import logger
def AssignBondOrdersFromTemplate(refmol, mol):
    """ assigns bond orders to a molecule based on the
    bond orders in a template molecule

    Arguments
      - refmol: the template molecule
      - mol: the molecule to assign bond orders to

    An example, start by generating a template from a SMILES
    and read in the PDB structure of the molecule

    >>> import os
    >>> from rdkit.Chem import AllChem
    >>> template = AllChem.MolFromSmiles("CN1C(=NC(C1=O)(c2ccccc2)c3ccccc3)N")
    >>> mol = AllChem.MolFromPDBFile(os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', '4DJU_lig.pdb'))
    >>> len([1 for b in template.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    8
    >>> len([1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    22

    Now assign the bond orders based on the template molecule

    >>> newMol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    >>> len([1 for b in newMol.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    8

    Note that the template molecule should have no explicit hydrogens
    else the algorithm will fail.

    It also works if there are different formal charges (this was github issue 235):

    >>> template=AllChem.MolFromSmiles('CN(C)C(=O)Cc1ccc2c(c1)NC(=O)c3ccc(cc3N2)c4ccc(c(c4)OC)[N+](=O)[O-]')
    >>> mol = AllChem.MolFromMolFile(os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', '4FTR_lig.mol'))
    >>> AllChem.MolToSmiles(mol)
    'COC1CC(C2CCC3C(O)NC4CC(CC(O)N(C)C)CCC4NC3C2)CCC1N(O)O'
    >>> newMol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    >>> AllChem.MolToSmiles(newMol)
    'COc1cc(-c2ccc3c(c2)Nc2ccc(CC(=O)N(C)C)cc2NC3=O)ccc1[N+](=O)[O-]'

    """
    refmol2 = rdchem.Mol(refmol)
    mol2 = rdchem.Mol(mol)
    matching = mol2.GetSubstructMatch(refmol2)
    if not matching:
        for b in mol2.GetBonds():
            if b.GetBondType() != BondType.SINGLE:
                b.SetBondType(BondType.SINGLE)
                b.SetIsAromatic(False)
        for b in refmol2.GetBonds():
            b.SetBondType(BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)
        matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
        if matching:
            if len(matching) > 1:
                logger.warning('More than one matching pattern found - picking one')
            matching = matching[0]
            for b in refmol.GetBonds():
                atom1 = matching[b.GetBeginAtomIdx()]
                atom2 = matching[b.GetEndAtomIdx()]
                b2 = mol2.GetBondBetweenAtoms(atom1, atom2)
                b2.SetBondType(b.GetBondType())
                b2.SetIsAromatic(b.GetIsAromatic())
            for a in refmol.GetAtoms():
                a2 = mol2.GetAtomWithIdx(matching[a.GetIdx()])
                a2.SetHybridization(a.GetHybridization())
                a2.SetIsAromatic(a.GetIsAromatic())
                a2.SetNumExplicitHs(a.GetNumExplicitHs())
                a2.SetFormalCharge(a.GetFormalCharge())
            SanitizeMol(mol2)
            if hasattr(mol2, '__sssAtoms'):
                mol2.__sssAtoms = None
        else:
            raise ValueError('No matching found')
    return mol2