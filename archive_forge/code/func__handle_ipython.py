from ._version import __version__, __protocol_version__, __jupyter_widgets_controls_version__, __jupyter_widgets_base_version__
import os
import sys
from traitlets import link, dlink
from IPython import get_ipython
from .widgets import *
def _handle_ipython():
    """Register with the comm target at import if running in Jupyter"""
    ip = get_ipython()
    if ip is None:
        return
    register_comm_target()