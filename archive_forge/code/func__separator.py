from .manage_plugins import *
from .sift import *
from .collection import *
from ._io import *
from ._image_stack import *
def _separator(char, lengths):
    return [char * separator_length for separator_length in lengths]