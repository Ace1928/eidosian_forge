import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis
def _get_gly_cb_vector(self, residue):
    """Return a pseudo CB vector for a Gly residue (PRIVATE).

        The pseudoCB vector is centered at the origin.

        CB coord=N coord rotated over -120 degrees
        along the CA-C axis.
        """
    try:
        n_v = residue['N'].get_vector()
        c_v = residue['C'].get_vector()
        ca_v = residue['CA'].get_vector()
    except Exception:
        return None
    n_v = n_v - ca_v
    c_v = c_v - ca_v
    rot = rotaxis(-pi * 120.0 / 180.0, c_v)
    cb_at_origin_v = n_v.left_multiply(rot)
    cb_v = cb_at_origin_v + ca_v
    self.ca_cb_list.append((ca_v, cb_v))
    return cb_at_origin_v