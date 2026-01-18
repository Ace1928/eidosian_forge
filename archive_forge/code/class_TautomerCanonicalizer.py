from warnings import warn
import copy
import logging
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondStereo, BondType
from .utils import memoized_property, pairwise
class TautomerCanonicalizer(object):
    """

    """

    def __init__(self, transforms=TAUTOMER_TRANSFORMS, scores=TAUTOMER_SCORES, max_tautomers=MAX_TAUTOMERS):
        """

        :param transforms: A list of TautomerTransforms to use to enumerate tautomers.
        :param scores: A list of TautomerScores to use to choose the canonical tautomer.
        :param max_tautomers: The maximum number of tautomers to enumerate, a limit to prevent combinatorial explosion.
        """
        self.transforms = transforms
        self.scores = scores
        self.max_tautomers = max_tautomers

    def __call__(self, mol):
        """Calling a TautomerCanonicalizer instance like a function is the same as calling its canonicalize(mol) method."""
        return self.canonicalize(mol)

    def canonicalize(self, mol):
        """Return a canonical tautomer by enumerating and scoring all possible tautomers.

        :param mol: The input molecule.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The canonical tautomer.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        tautomers = self._enumerate_tautomers(mol)
        if len(tautomers) == 1:
            return tautomers[0]
        highest = None
        for t in tautomers:
            smiles = Chem.MolToSmiles(t, isomericSmiles=True)
            log.debug(f'Tautomer: {smiles}')
            score = 0
            ssr = Chem.GetSymmSSSR(t)
            for ring in ssr:
                btypes = {t.GetBondBetweenAtoms(*pair).GetBondType() for pair in pairwise(ring)}
                elements = {t.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring}
                if btypes == {BondType.AROMATIC}:
                    log.debug('Score +100 (aromatic ring)')
                    score += 100
                    if elements == {6}:
                        log.debug('Score +150 (carbocyclic aromatic ring)')
                        score += 150
            for tscore in self.scores:
                for _ in t.GetSubstructMatches(tscore.smarts):
                    log.debug('Score %+d (%s)', tscore.score, tscore.name)
                    score += tscore.score
            for atom in t.GetAtoms():
                if atom.GetAtomicNum() in {15, 16, 34, 52}:
                    hs = atom.GetTotalNumHs()
                    if hs:
                        log.debug('Score %+d (%s-H bonds)', -hs, atom.GetSymbol())
                        score -= hs
            if not highest or highest['score'] < score or (highest['score'] == score and smiles < highest['smiles']):
                log.debug(f'New highest tautomer: {smiles} ({score})')
                highest = {'smiles': smiles, 'tautomer': t, 'score': score}
        return highest['tautomer']

    @memoized_property
    def _enumerate_tautomers(self):
        return TautomerEnumerator(self.transforms, self.max_tautomers)