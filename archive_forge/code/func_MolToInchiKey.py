import logging
from rdkit import RDLogger
from rdkit.Chem import rdinchi
def MolToInchiKey(mol, options=''):
    """Returns the standard InChI key for a molecule

    Returns:
    the standard InChI key returned by InChI API for the input molecule
    """
    return rdinchi.MolToInchiKey(mol, options)