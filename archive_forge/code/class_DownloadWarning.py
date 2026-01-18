import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
class DownloadWarning(Warning):
    """Issued when a file is being downloaded by a :class:`Downloader`."""
    pass