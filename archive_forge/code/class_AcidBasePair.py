from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
class AcidBasePair(object):
    """An acid and its conjugate base, defined by SMARTS.

    A strength-ordered list of AcidBasePairs can be used to ensure the strongest acids in a molecule ionize first.
    """

    def __init__(self, name, acid, base):
        """Initialize an AcidBasePair with the following parameters:

        :param string name: A name for this AcidBasePair.
        :param string acid: SMARTS pattern for the protonated acid.
        :param string base: SMARTS pattern for the conjugate ionized base.
        """
        log.debug(f'Initializing AcidBasePair: {name}')
        self.name = name
        self.acid_str = acid
        self.base_str = base

    @memoized_property
    def acid(self):
        log.debug(f'Loading AcidBasePair acid: {self.name}')
        return Chem.MolFromSmarts(self.acid_str)

    @memoized_property
    def base(self):
        log.debug(f'Loading AcidBasePair base: {self.name}')
        return Chem.MolFromSmarts(self.base_str)

    def __repr__(self):
        return 'AcidBasePair({!r}, {!r}, {!r})'.format(self.name, self.acid_str, self.base_str)

    def __str__(self):
        return self.name