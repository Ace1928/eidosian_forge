import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis
class HSExposureCA(_AbstractHSExposure):
    """Class to calculate HSE based on the approximate CA-CB vectors.

    Uses three consecutive CA positions.
    """

    def __init__(self, model, radius=12, offset=0):
        """Initialize class.

        :param model: the model that contains the residues
        :type model: L{Model}

        :param radius: radius of the sphere (centred at the CA atom)
        :type radius: float

        :param offset: number of flanking residues that are ignored
                       in the calculation of the number of neighbors
        :type offset: int
        """
        _AbstractHSExposure.__init__(self, model, radius, offset, 'EXP_HSE_A_U', 'EXP_HSE_A_D', 'EXP_CB_PCB_ANGLE')

    def _get_cb(self, r1, r2, r3):
        """Calculate approx CA-CB direction (PRIVATE).

        Calculate the approximate CA-CB direction for a central
        CA atom based on the two flanking CA positions, and the angle
        with the real CA-CB vector.

        The CA-CB vector is centered at the origin.

        :param r1, r2, r3: three consecutive residues
        :type r1, r2, r3: L{Residue}
        """
        if r1 is None or r3 is None:
            return None
        try:
            ca1 = r1['CA'].get_vector()
            ca2 = r2['CA'].get_vector()
            ca3 = r3['CA'].get_vector()
        except Exception:
            return None
        d1 = ca2 - ca1
        d3 = ca2 - ca3
        d1.normalize()
        d3.normalize()
        b = d1 + d3
        b.normalize()
        self.ca_cb_list.append((ca2, b + ca2))
        if r2.has_id('CB'):
            cb = r2['CB'].get_vector()
            cb_ca = cb - ca2
            cb_ca.normalize()
            angle = cb_ca.angle(b)
        elif r2.get_resname() == 'GLY':
            cb_ca = self._get_gly_cb_vector(r2)
            if cb_ca is None:
                angle = None
            else:
                angle = cb_ca.angle(b)
        else:
            angle = None
        return (b, angle)

    def pcb_vectors_pymol(self, filename='hs_exp.py'):
        """Write PyMol script for visualization.

        Write a PyMol script that visualizes the pseudo CB-CA directions
        at the CA coordinates.

        :param filename: the name of the pymol script file
        :type filename: string
        """
        if not self.ca_cb_list:
            warnings.warn('Nothing to draw.', RuntimeWarning)
            return
        with open(filename, 'w') as fp:
            fp.write('from pymol.cgo import *\n')
            fp.write('from pymol import cmd\n')
            fp.write('obj=[\n')
            fp.write('BEGIN, LINES,\n')
            fp.write(f'COLOR, {1.0:.2f}, {1.0:.2f}, {1.0:.2f},\n')
            for ca, cb in self.ca_cb_list:
                x, y, z = ca.get_array()
                fp.write(f'VERTEX, {x:.2f}, {y:.2f}, {z:.2f},\n')
                x, y, z = cb.get_array()
                fp.write(f'VERTEX, {x:.2f}, {y:.2f}, {z:.2f},\n')
            fp.write('END]\n')
            fp.write("cmd.load_cgo(obj, 'HS')\n")