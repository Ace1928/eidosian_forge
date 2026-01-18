from Bio.PDB.Entity import Entity
Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems

        :raises Exception: if any chain does not have .internal_coord attribute
        