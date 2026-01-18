import logging
import re
from rdkit import Chem
from rdkit.Chem import inchi
def _is_achiral_by_symmetry(INCHI):
    mol = Chem.MolFromInchi(INCHI)
    if not mol:
        mol = Chem.MolFromInchi(f'InChI=1/{INCHI}')
    try:
        return len(Chem.FindMolChiralCenters(mol, True, True)) == 0
    except Exception:
        return False