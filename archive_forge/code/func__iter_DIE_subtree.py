from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def _iter_DIE_subtree(self, die):
    """ Given a DIE, this yields it with its subtree including null DIEs
            (child list terminators).
        """
    if die.tag == 'DW_TAG_imported_unit' and self.dwarfinfo.supplementary_dwarfinfo:
        die = die.get_DIE_from_attribute('DW_AT_import')
    yield die
    if die.has_children:
        for c in die.iter_children():
            for d in die.cu._iter_DIE_subtree(c):
                yield d
        yield die._terminator