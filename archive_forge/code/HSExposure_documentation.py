import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis
Initialize class.

        A residue's exposure is defined as the number of CA atoms around
        that residue's CA atom. A dictionary is returned that uses a L{Residue}
        object as key, and the residue exposure as corresponding value.

        :param model: the model that contains the residues
        :type model: L{Model}

        :param radius: radius of the sphere (centred at the CA atom)
        :type radius: float

        :param offset: number of flanking residues that are ignored in
                       the calculation of the number of neighbors
        :type offset: int

        