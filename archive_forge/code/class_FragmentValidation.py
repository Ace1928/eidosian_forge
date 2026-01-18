from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
class FragmentValidation(Validation):
    """Logs if certain fragments are present.

    Subclass and override the ``fragments`` class attribute to customize the list of
    :class:`FragmentPatterns <molvs.fragment.FragmentPattern>`.
    """
    fragments = REMOVE_FRAGMENTS

    def run(self, mol):
        for fp in self.fragments:
            matches = frozenset((frozenset(match) for match in mol.GetSubstructMatches(fp.smarts)))
            fragments = frozenset((frozenset(frag) for frag in Chem.GetMolFrags(mol)))
            if matches & fragments:
                self.log.info(f'{fp.name} is present')