from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def _set_flag(self, flag):
    """
        Set the given flag or flags.

        :param int flag: flag to set; may be OR'd combination of flags
        """
    self.conflags |= flag