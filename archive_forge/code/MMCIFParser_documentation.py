import numpy as np
import warnings
from Bio.File import as_handle
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
Return the structure.

        Arguments:
         - structure_id - string, the id that will be used for the structure
         - filename - name of the mmCIF file OR an open filehandle

        