from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def _inherit_flags(self, *subcons):
    """
        Pull flags from subconstructs.
        """
    for sc in subcons:
        self._set_flag(sc.conflags)