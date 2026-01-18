from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
Download a .zip file from a URL and extract it.

    Parameters
    ----------
    url : string
        URL of file to be downloaded
    destination : string
        Directory in which to extract the downloaded zip
    