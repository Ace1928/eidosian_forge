import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def abspath_forward_slashes(path):
    """Return forward-slashed version of os.path.abspath"""
    return '/'.join(os.path.abspath(path).split(os.path.sep))