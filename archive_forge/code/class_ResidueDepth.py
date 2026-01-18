import os
import subprocess
import tempfile
import warnings
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import is_aa
from Bio import BiopythonWarning
class ResidueDepth(AbstractPropertyMap):
    """Calculate residue and CA depth for all residues."""

    def __init__(self, model, msms_exec=None):
        """Initialize the class."""
        if msms_exec is None:
            msms_exec = 'msms'
        depth_dict = {}
        depth_list = []
        depth_keys = []
        residue_list = Selection.unfold_entities(model, 'R')
        surface = get_surface(model, MSMS=msms_exec)
        for residue in residue_list:
            if not is_aa(residue):
                continue
            rd = residue_depth(residue, surface)
            ca_rd = ca_depth(residue, surface)
            res_id = residue.get_id()
            chain_id = residue.get_parent().get_id()
            depth_dict[chain_id, res_id] = (rd, ca_rd)
            depth_list.append((residue, (rd, ca_rd)))
            depth_keys.append((chain_id, res_id))
            residue.xtra['EXP_RD'] = rd
            residue.xtra['EXP_RD_CA'] = ca_rd
        AbstractPropertyMap.__init__(self, depth_dict, depth_keys, depth_list)