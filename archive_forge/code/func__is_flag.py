from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def _is_flag(self, flag):
    """
        Check whether a given flag is set.

        :param int flag: flag to check
        """
    return bool(self.conflags & flag)