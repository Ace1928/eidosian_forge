import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
Manage extra help entries.
        Pass the labels of the extra commands to be activated.
        