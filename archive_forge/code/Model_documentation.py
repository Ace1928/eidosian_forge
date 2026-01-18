from Bio.PDB.Entity import Entity
from Bio.PDB.internal_coords import IC_Chain
Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems

        :raises Exception: if any chain does not have .pic attribute
        