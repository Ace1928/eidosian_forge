import collections
import math
import numpy as np
from Bio.PDB.kdtrees import KDTree
Calculate surface accessibility surface area for an entity.

        The resulting atomic surface accessibility values are attached to the
        .sasa attribute of each entity (or atom), depending on the level. For
        example, if level="R", all residues will have a .sasa attribute. Atoms
        will always be assigned a .sasa attribute with their individual values.

        :param entity: input entity.
        :type entity: Bio.PDB.Entity, e.g. Residue, Chain, ...

        :param level: the level at which ASA values are assigned, which can be
            one of "A" (Atom), "R" (Residue), "C" (Chain), "M" (Model), or
            "S" (Structure). The ASA value of an entity is the sum of all ASA
            values of its children. Defaults to "A".
        :type entity: Bio.PDB.Entity

        >>> from Bio.PDB import PDBParser
        >>> from Bio.PDB.SASA import ShrakeRupley
        >>> p = PDBParser(QUIET=1)
        >>> # This assumes you have a local copy of 1LCD.pdb in a directory called "PDB"
        >>> struct = p.get_structure("1LCD", "PDB/1LCD.pdb")
        >>> sr = ShrakeRupley()
        >>> sr.compute(struct, level="S")
        >>> print(round(struct.sasa, 2))
        7053.43
        >>> print(round(struct[0]["A"][11]["OE1"].sasa, 2))
        9.64
        