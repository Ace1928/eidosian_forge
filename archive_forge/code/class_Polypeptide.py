import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
class Polypeptide(list):
    """A polypeptide is simply a list of L{Residue} objects."""

    def get_ca_list(self):
        """Get list of C-alpha atoms in the polypeptide.

        :return: the list of C-alpha atoms
        :rtype: [L{Atom}, L{Atom}, ...]
        """
        ca_list = []
        for res in self:
            ca = res['CA']
            ca_list.append(ca)
        return ca_list

    def get_phi_psi_list(self):
        """Return the list of phi/psi dihedral angles."""
        ppl = []
        lng = len(self)
        for i in range(lng):
            res = self[i]
            try:
                n = res['N'].get_vector()
                ca = res['CA'].get_vector()
                c = res['C'].get_vector()
            except Exception:
                ppl.append((None, None))
                res.xtra['PHI'] = None
                res.xtra['PSI'] = None
                continue
            if i > 0:
                rp = self[i - 1]
                try:
                    cp = rp['C'].get_vector()
                    phi = calc_dihedral(cp, n, ca, c)
                except Exception:
                    phi = None
            else:
                phi = None
            if i < lng - 1:
                rn = self[i + 1]
                try:
                    nn = rn['N'].get_vector()
                    psi = calc_dihedral(n, ca, c, nn)
                except Exception:
                    psi = None
            else:
                psi = None
            ppl.append((phi, psi))
            res.xtra['PHI'] = phi
            res.xtra['PSI'] = psi
        return ppl

    def get_tau_list(self):
        """List of tau torsions angles for all 4 consecutive Calpha atoms."""
        ca_list = self.get_ca_list()
        tau_list = []
        for i in range(len(ca_list) - 3):
            atom_list = (ca_list[i], ca_list[i + 1], ca_list[i + 2], ca_list[i + 3])
            v1, v2, v3, v4 = (a.get_vector() for a in atom_list)
            tau = calc_dihedral(v1, v2, v3, v4)
            tau_list.append(tau)
            res = ca_list[i + 2].get_parent()
            res.xtra['TAU'] = tau
        return tau_list

    def get_theta_list(self):
        """List of theta angles for all 3 consecutive Calpha atoms."""
        theta_list = []
        ca_list = self.get_ca_list()
        for i in range(len(ca_list) - 2):
            atom_list = (ca_list[i], ca_list[i + 1], ca_list[i + 2])
            v1, v2, v3 = (a.get_vector() for a in atom_list)
            theta = calc_angle(v1, v2, v3)
            theta_list.append(theta)
            res = ca_list[i + 1].get_parent()
            res.xtra['THETA'] = theta
        return theta_list

    def get_sequence(self):
        """Return the AA sequence as a Seq object.

        :return: polypeptide sequence
        :rtype: L{Seq}
        """
        s = ''.join((protein_letters_3to1_extended.get(res.get_resname(), 'X') for res in self))
        return Seq(s)

    def __repr__(self):
        """Return string representation of the polypeptide.

        Return <Polypeptide start=START end=END>, where START
        and END are sequence identifiers of the outer residues.
        """
        start = self[0].get_id()[1]
        end = self[-1].get_id()[1]
        return f'<Polypeptide start={start} end={end}>'