import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def SnapPea_callback(self, interrupted=False):
    """
        Callback for SnapPea to keep the UI alive during long computations.
        """
    if interrupted:
        self.interrupted = False
        raise KeyboardInterrupt('SnapPea computation aborted')