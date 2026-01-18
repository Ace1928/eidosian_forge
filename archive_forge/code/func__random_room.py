import random
from IPython.display import display
import ipywidgets as widgets
from ._version import version_info, __version__  # noqa
from .webrtc import *  # noqa
def _random_room():
    return ''.join((chr(ord('0') + random.randint(0, 9)) for k in range(6)))