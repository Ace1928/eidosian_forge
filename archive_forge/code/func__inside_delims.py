import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@classmethod
def _inside_delims(cls, line, opener, closer):
    return (line.count(opener) - line.count(closer)) % 2