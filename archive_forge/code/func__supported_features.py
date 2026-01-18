import json
import os
import os.path
import re
import shutil
import sys
import traceback
from glob import glob
from importlib import import_module
from os.path import join as pjoin
def _supported_features():
    """Return the list of options features supported by the backend.

    Returns a list of strings.
    The only possible value is 'build_editable'.
    """
    backend = _build_backend()
    features = []
    if hasattr(backend, 'build_editable'):
        features.append('build_editable')
    return features