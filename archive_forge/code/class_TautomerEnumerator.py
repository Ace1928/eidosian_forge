from warnings import warn
import copy
import logging
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondStereo, BondType
from .utils import memoized_property, pairwise
class TautomerEnumerator(object):
    """

    """

    def __init__(self, transforms=TAUTOMER_TRANSFORMS, max_tautomers=MAX_TAUTOMERS):
        """

        :param transforms: A list of TautomerTransforms to use to enumerate tautomers.
        :param max_tautomers: The maximum number of tautomers to enumerate (limit to prevent combinatorial explosion).
        """
        self.transforms = transforms
        self.max_tautomers = max_tautomers

    def __call__(self, mol):
        """Calling a TautomerEnumerator instance like a function is the same as calling its enumerate(mol) method."""
        return self.enumerate(mol)

    def enumerate(self, mol):
        """Enumerate all possible tautomers and return them as a list.

        :param mol: The input molecule.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: A list of all possible tautomers of the molecule.
        :rtype: list of :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        tautomers = {smiles: copy.deepcopy(mol)}
        kekulized = copy.deepcopy(mol)
        Chem.Kekulize(kekulized)
        kekulized = {smiles: kekulized}
        done = set()
        while len(tautomers) < self.max_tautomers:
            for tsmiles in sorted(tautomers):
                if tsmiles in done:
                    continue
                for transform in self.transforms:
                    for match in kekulized[tsmiles].GetSubstructMatches(transform.tautomer):
                        product = copy.deepcopy(kekulized[tsmiles])
                        first = product.GetAtomWithIdx(match[0])
                        last = product.GetAtomWithIdx(match[-1])
                        first.SetNumExplicitHs(max(0, first.GetTotalNumHs() - 1))
                        last.SetNumExplicitHs(last.GetTotalNumHs() + 1)
                        first.SetNoImplicit(True)
                        last.SetNoImplicit(True)
                        for bi, pair in enumerate(pairwise(match)):
                            if transform.bonds:
                                product.GetBondBetweenAtoms(*pair).SetBondType(transform.bonds[bi])
                            else:
                                current_bond_type = product.GetBondBetweenAtoms(*pair).GetBondType()
                                product.GetBondBetweenAtoms(*pair).SetBondType(BondType.DOUBLE if current_bond_type == BondType.SINGLE else BondType.SINGLE)
                        if transform.charges:
                            for ci, idx in enumerate(match):
                                atom = product.GetAtomWithIdx(idx)
                                atom.SetFormalCharge(atom.GetFormalCharge() + transform.charges[ci])
                        try:
                            Chem.SanitizeMol(product)
                            smiles = Chem.MolToSmiles(product, isomericSmiles=True)
                            log.debug(f'Applied rule: {transform.name} to {tsmiles}')
                            if smiles not in tautomers:
                                log.debug(f'New tautomer produced: {smiles}')
                                kekulized_product = copy.deepcopy(product)
                                Chem.Kekulize(kekulized_product)
                                tautomers[smiles] = product
                                kekulized[smiles] = kekulized_product
                            else:
                                log.debug(f'Previous tautomer produced again: {smiles}')
                        except ValueError:
                            log.debug(f'ValueError Applying rule: {transform.name}')
                done.add(tsmiles)
            if len(tautomers) == len(done):
                break
        else:
            log.warning(f'Tautomer enumeration stopped at maximum {self.max_tautomers}')
        for tautomer in tautomers.values():
            Chem.AssignStereochemistry(tautomer, force=True, cleanIt=True)
            for bond in tautomer.GetBonds():
                if bond.GetBondType() == BondType.DOUBLE and bond.GetStereo() > BondStereo.STEREOANY:
                    begin = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    for othertautomer in tautomers.values():
                        if not othertautomer.GetBondBetweenAtoms(begin, end).GetBondType() == BondType.DOUBLE:
                            neighbours = tautomer.GetAtomWithIdx(begin).GetBonds() + tautomer.GetAtomWithIdx(end).GetBonds()
                            for otherbond in neighbours:
                                if otherbond.GetBondDir() in {BondDir.ENDUPRIGHT, BondDir.ENDDOWNRIGHT}:
                                    otherbond.SetBondDir(BondDir.NONE)
                            Chem.AssignStereochemistry(tautomer, force=True, cleanIt=True)
                            log.debug('Removed stereochemistry from unfixed double bond')
                            break
        return list(tautomers.values())