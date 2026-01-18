import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class DistanceMatrix(_Matrix):
    """Distance matrix class that can be used for distance based tree algorithms.

    All diagonal elements will be zero no matter what the users provide.
    """

    def __init__(self, names, matrix=None):
        """Initialize the class."""
        _Matrix.__init__(self, names, matrix)
        self._set_zero_diagonal()

    def __setitem__(self, item, value):
        """Set Matrix's items to values."""
        _Matrix.__setitem__(self, item, value)
        self._set_zero_diagonal()

    def _set_zero_diagonal(self):
        """Set all diagonal elements to zero (PRIVATE)."""
        for i in range(0, len(self)):
            self.matrix[i][i] = 0

    def format_phylip(self, handle):
        """Write data in Phylip format to a given file-like object or handle.

        The output stream is the input distance matrix format used with Phylip
        programs (e.g. 'neighbor'). See:
        http://evolution.genetics.washington.edu/phylip/doc/neighbor.html

        :Parameters:
            handle : file or file-like object
                A writeable text mode file handle or other object supporting
                the 'write' method, such as StringIO or sys.stdout.

        """
        handle.write(f'    {len(self.names)}\n')
        name_width = max(12, max(map(len, self.names)) + 1)
        value_fmts = ('{' + str(x) + ':.4f}' for x in range(1, len(self.matrix) + 1))
        row_fmt = '{0:' + str(name_width) + 's}' + '  '.join(value_fmts) + '\n'
        for i, (name, values) in enumerate(zip(self.names, self.matrix)):
            mirror_values = (self.matrix[j][i] for j in range(i + 1, len(self.matrix)))
            fields = itertools.chain([name], values, mirror_values)
            handle.write(row_fmt.format(*fields))