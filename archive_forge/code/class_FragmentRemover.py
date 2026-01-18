from warnings import warn
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .utils import memoized_property
class FragmentRemover(object):
    """A class for filtering out fragments using SMARTS patterns."""

    def __init__(self, fragments=REMOVE_FRAGMENTS, leave_last=LEAVE_LAST):
        """Initialize a FragmentRemover with an optional custom list of :class:`~molvs.fragment.FragmentPattern`.

        Setting leave_last to True will ensure at least one fragment is left in the molecule, even if it is matched by a
        :class:`~molvs.fragment.FragmentPattern`. Fragments are removed in the order specified in the list, so place
        those you would prefer to be left towards the end of the list. If all the remaining fragments match the same
        :class:`~molvs.fragment.FragmentPattern`, they will all be left.

        :param fragments: A list of :class:`~molvs.fragment.FragmentPattern` to remove.
        :param bool leave_last: Whether to ensure at least one fragment is left.
        """
        log.debug('Initializing FragmentRemover')
        self.fragments = fragments
        self.leave_last = leave_last

    def __call__(self, mol):
        """Calling a FragmentRemover instance like a function is the same as calling its remove(mol) method."""
        return self.remove(mol)

    def remove(self, mol):
        """Return the molecule with specified fragments removed.

        :param mol: The molecule to remove fragments from.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The molecule with fragments removed.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        log.debug('Running FragmentRemover')
        for frag in self.fragments:
            if mol.GetNumAtoms() == 0 or (self.leave_last and len(Chem.GetMolFrags(mol)) <= 1):
                break
            removed = Chem.DeleteSubstructs(mol, frag.smarts, onlyFrags=True)
            if mol.GetNumAtoms() != removed.GetNumAtoms():
                log.info(f'Removed fragment: {frag.name}')
            if self.leave_last and removed.GetNumAtoms() == 0:
                break
            mol = removed
        return mol