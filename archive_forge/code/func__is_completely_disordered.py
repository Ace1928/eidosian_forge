import warnings
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
def _is_completely_disordered(self, residue):
    """Return 1 if all atoms in the residue have a non blank altloc (PRIVATE)."""
    atom_list = residue.get_unpacked_list()
    for atom in atom_list:
        altloc = atom.get_altloc()
        if altloc == ' ':
            return 0
    return 1