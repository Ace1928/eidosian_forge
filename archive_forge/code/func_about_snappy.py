import sys
import os
import datetime
from .gui import *
from .version import version as SnapPy_version
from IPython import __version__ as IPython_version
def about_snappy(window):
    return InfoWindow(window, 'About SnapPy', about_snappy_text, 'about_snappy')