from warnings import warn
import copy
import logging
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondStereo, BondType
from .utils import memoized_property, pairwise
class TautomerScore(object):
    """A substructure defined by SMARTS and its score contribution to determine the canonical tautomer."""

    def __init__(self, name, smarts, score):
        """Initialize a TautomerScore with a name, SMARTS pattern and score.

        :param name: A name for this TautomerScore.
        :param smarts: SMARTS pattern to match a substructure.
        :param score: The score to assign for this substructure.
        """
        self.name = name
        self.smarts_str = smarts
        self.score = score

    @memoized_property
    def smarts(self):
        return Chem.MolFromSmarts(self.smarts_str)

    def __repr__(self):
        return 'TautomerScore({!r}, {!r}, {!r})'.format(self.name, self.smarts_str, self.score)

    def __str__(self):
        return self.name