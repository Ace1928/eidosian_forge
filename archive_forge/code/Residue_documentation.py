from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
Remove a child residue from the DisorderedResidue.

        Arguments:
         - resname - name of the child residue to remove, as a string.

        