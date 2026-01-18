from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def _clear_flag(self, flag):
    """
        Clear the given flag or flags.

        :param int flag: flag to clear; may be OR'd combination of flags
        """
    self.conflags &= ~flag