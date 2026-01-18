import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
class DiffIndex(List[T_Diff]):
    """An Index for diffs, allowing a list of Diffs to be queried by the diff
    properties.

    The class improves the diff handling convenience.
    """
    change_type = ('A', 'C', 'D', 'R', 'M', 'T')
    'Change type invariant identifying possible ways a blob can have changed:\n\n    * ``A`` = Added\n    * ``D`` = Deleted\n    * ``R`` = Renamed\n    * ``M`` = Modified\n    * ``T`` = Changed in the type\n    '

    def iter_change_type(self, change_type: Lit_change_type) -> Iterator[T_Diff]:
        """
        :return:
            Iterator yielding :class:`Diff` instances that match the given `change_type`

        :param change_type:
            Member of :attr:`DiffIndex.change_type`, namely:

            * 'A' for added paths
            * 'D' for deleted paths
            * 'R' for renamed paths
            * 'M' for paths with modified data
            * 'T' for changed in the type paths
        """
        if change_type not in self.change_type:
            raise ValueError('Invalid change type: %s' % change_type)
        for diffidx in self:
            if diffidx.change_type == change_type:
                yield diffidx
            elif change_type == 'A' and diffidx.new_file:
                yield diffidx
            elif change_type == 'D' and diffidx.deleted_file:
                yield diffidx
            elif change_type == 'C' and diffidx.copied_file:
                yield diffidx
            elif change_type == 'R' and diffidx.renamed:
                yield diffidx
            elif change_type == 'M' and diffidx.a_blob and diffidx.b_blob and (diffidx.a_blob != diffidx.b_blob):
                yield diffidx