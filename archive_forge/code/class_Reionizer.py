from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
class Reionizer(object):
    """A class to fix charges and reionize a molecule such that the strongest acids ionize first."""

    def __init__(self, acid_base_pairs=ACID_BASE_PAIRS, charge_corrections=CHARGE_CORRECTIONS):
        """Initialize a Reionizer with the following parameter:

        :param acid_base_pairs: A list of :class:`AcidBasePairs <molvs.charge.AcidBasePair>` to reionize, sorted from
                                strongest to weakest.
        :param charge_corrections: A list of :class:`ChargeCorrections <molvs.charge.ChargeCorrection>`.
        """
        log.debug('Initializing Reionizer')
        self.acid_base_pairs = acid_base_pairs
        self.charge_corrections = charge_corrections

    def __call__(self, mol):
        """Calling a Reionizer instance like a function is the same as calling its reionize(mol) method."""
        return self.reionize(mol)

    def reionize(self, mol):
        """Enforce charges on certain atoms, then perform competitive reionization.

        First, charge corrections are applied to ensure, for example, that free metals are correctly ionized. Then, if
        a molecule with multiple acid groups is partially ionized, ensure the strongest acids ionize first.

        The algorithm works as follows:

        - Use SMARTS to find the strongest protonated acid and the weakest ionized acid.
        - If the ionized acid is weaker than the protonated acid, swap proton and repeat.

        :param mol: The molecule to reionize.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The reionized molecule.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        log.debug('Running Reionizer')
        start_charge = Chem.GetFormalCharge(mol)
        for cc in self.charge_corrections:
            for match in mol.GetSubstructMatches(cc.smarts):
                atom = mol.GetAtomWithIdx(match[0])
                log.info('Applying charge correction %s (%s %+d)', cc.name, atom.GetSymbol(), cc.charge)
                atom.SetFormalCharge(cc.charge)
        current_charge = Chem.GetFormalCharge(mol)
        charge_diff = Chem.GetFormalCharge(mol) - start_charge
        if not current_charge == 0:
            while charge_diff > 0:
                ppos, poccur = self._strongest_protonated(mol)
                if ppos is None:
                    break
                log.info(f'Ionizing {self.acid_base_pairs[ppos].name} to balance previous charge corrections')
                patom = mol.GetAtomWithIdx(poccur[-1])
                patom.SetFormalCharge(patom.GetFormalCharge() - 1)
                if patom.GetNumExplicitHs() > 0:
                    patom.SetNumExplicitHs(patom.GetNumExplicitHs() - 1)
                patom.UpdatePropertyCache()
                charge_diff -= 1
        already_moved = set()
        while True:
            ppos, poccur = self._strongest_protonated(mol)
            ipos, ioccur = self._weakest_ionized(mol)
            if ioccur and poccur and (ppos < ipos):
                if poccur[-1] == ioccur[-1]:
                    log.warning('Aborted reionization due to unexpected situation')
                    break
                key = tuple(sorted([poccur[-1], ioccur[-1]]))
                if key in already_moved:
                    log.warning('Aborting reionization to avoid infinite loop due to it being ambiguous where to put a Hydrogen')
                    break
                already_moved.add(key)
                log.info(f'Moved proton from {self.acid_base_pairs[ppos].name} to {self.acid_base_pairs[ipos].name}')
                patom = mol.GetAtomWithIdx(poccur[-1])
                patom.SetFormalCharge(patom.GetFormalCharge() - 1)
                if patom.GetNumImplicitHs() == 0 and patom.GetNumExplicitHs() > 0:
                    patom.SetNumExplicitHs(patom.GetNumExplicitHs() - 1)
                patom.UpdatePropertyCache()
                iatom = mol.GetAtomWithIdx(ioccur[-1])
                iatom.SetFormalCharge(iatom.GetFormalCharge() + 1)
                if iatom.GetNoImplicit() or ((patom.GetAtomicNum() == 7 or patom.GetAtomicNum() == 15) and patom.GetIsAromatic()) or iatom.GetTotalValence() not in list(Chem.GetPeriodicTable().GetValenceList(iatom.GetAtomicNum())):
                    iatom.SetNumExplicitHs(iatom.GetNumExplicitHs() + 1)
                iatom.UpdatePropertyCache()
            else:
                break
        Chem.SanitizeMol(mol)
        return mol

    def _strongest_protonated(self, mol):
        for position, pair in enumerate(self.acid_base_pairs):
            for occurrence in mol.GetSubstructMatches(pair.acid):
                return (position, occurrence)
        return (None, None)

    def _weakest_ionized(self, mol):
        for position, pair in enumerate(reversed(self.acid_base_pairs)):
            for occurrence in mol.GetSubstructMatches(pair.base):
                return (len(self.acid_base_pairs) - position - 1, occurrence)
        return (None, None)