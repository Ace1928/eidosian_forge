from warnings import warn
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .utils import memoized_property
class FragmentPattern(object):
    """A fragment defined by a SMARTS pattern."""

    def __init__(self, name, smarts):
        """Initialize a FragmentPattern with a name and a SMARTS pattern.

        :param name: A name for this FragmentPattern.
        :param smarts: A SMARTS pattern.
        """
        self.name = name
        self.smarts_str = smarts

    @memoized_property
    def smarts(self):
        return Chem.MolFromSmarts(self.smarts_str)

    def __repr__(self):
        return 'FragmentPattern({!r}, {!r})'.format(self.name, self.smarts_str)

    def __str__(self):
        return self.name