from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
class _SpacialScore:
    """Class intended for calculating spacial score (SPS) and size-normalised SPS (nSPS) for small organic molecules"""

    def __init__(self, mol, normalize=True):
        if mol is None:
            raise ValueError('No valid molecule object found.')
        molCp = Chem.Mol(mol)
        rdmolops.FindPotentialStereoBonds(molCp)
        self.mol = molCp
        self.normalize = normalize
        self.hyb_score = {}
        self.stereo_score = {}
        self.ring_score = {}
        self.bond_score = {}
        self.chiral_idxs = self.findStereoAtomIdxs()
        self.doublebonds_stereo = self.findDoubleBondsStereo()
        self.score = self.calculateSpacialScore()
        if normalize:
            self.score /= self.mol.GetNumHeavyAtoms()

    def findStereoAtomIdxs(self, includeUnassigned=True):
        """Finds indices of atoms that are (pseudo)stereo/chiralcentres, in respect to the attached groups (does not account for double bond isomers)"""
        stereo_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=includeUnassigned, includeCIP=False, useLegacyImplementation=False)
        stereo_idxs = [atom_idx for atom_idx, _ in stereo_centers]
        return stereo_idxs

    def findDoubleBondsStereo(self):
        """Finds indeces of stereo double bond atoms (E/Z)"""
        db_stereo = {}
        for bond in self.mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                db_stereo[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetStereo()
        return db_stereo

    def calculateSpacialScore(self):
        """Calculates the total spacial score for a molecule"""
        score = 0
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            self.hyb_score[atom_idx] = self._accountForHybridisation(atom)
            self.stereo_score[atom_idx] = self._accountForStereo(atom_idx)
            self.ring_score[atom_idx] = self._accountForRing(atom)
            self.bond_score[atom_idx] = self._accountForNeighbors(atom)
            score += self._calculateScoreForAtom(atom_idx)
        return score

    def _calculateScoreForAtom(self, atom_idx):
        """Calculates the total score for a single atom in a molecule"""
        atom_score = self.hyb_score[atom_idx] * self.stereo_score[atom_idx] * self.ring_score[atom_idx] * self.bond_score[atom_idx]
        return atom_score
    _hybridisations = defaultdict(lambda: 4)
    _hybridisations.update({Chem.HybridizationType.SP: 1, Chem.HybridizationType.SP2: 2, Chem.HybridizationType.SP3: 3})

    def _accountForHybridisation(self, atom):
        """Calculates the hybridisation score for a single atom in a molecule"""
        return self._hybridisations[atom.GetHybridization()]

    def _accountForStereo(self, atom_idx):
        """Calculates the stereo score for a single atom in a molecule"""
        if atom_idx in self.chiral_idxs:
            return 2
        for bond_atom_idxs, stereo in self.doublebonds_stereo.items():
            if stereo != Chem.BondStereo.STEREONONE and atom_idx in bond_atom_idxs:
                return 2
        return 1

    def _accountForRing(self, atom):
        """Calculates the ring score for a single atom in a molecule"""
        if atom.GetIsAromatic():
            return 1
        if atom.IsInRing():
            return 2
        return 1

    def _accountForNeighbors(self, atom):
        """Calculates the neighbour score for a single atom in a molecule
        The second power allows to account for branching in the molecular structure"""
        return atom.GetDegree() ** 2