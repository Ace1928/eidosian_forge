from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def _search_ancestor_offspring(self):
    """ Search our ancestors identifying their offspring to find our parent.

            DIEs are stored as a flattened tree.  The top DIE is the ancestor
            of all DIEs in the unit.  Each parent is guaranteed to be at
            an offset less than their children.  In each generation of children
            the sibling with the closest offset not greater than our offset is
            our ancestor.
        """
    search = self.cu.get_top_DIE()
    while search.offset < self.offset:
        prev = search
        for child in search.iter_children():
            child.set_parent(search)
            if child.offset <= self.offset:
                prev = child
        if search.has_children and search._terminator.offset <= self.offset:
            prev = search._terminator
        if prev is search:
            raise ValueError('offset %s not in CU %s DIE tree' % (self.offset, self.cu.cu_offset))
        search = prev