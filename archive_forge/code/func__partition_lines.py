import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@classmethod
def _partition_lines(cls, line, cell):
    """
        Check the code for additional use of %%opts. Enables
        multi-line use of %%opts in a single call to the magic.
        """
    if cell is None:
        return (line, cell)
    specs, code = ([line], [])
    for line in cell.splitlines():
        if line.strip().startswith('%%opts'):
            specs.append(line.strip()[7:])
        else:
            code.append(line)
    return (' '.join(specs), '\n'.join(code))