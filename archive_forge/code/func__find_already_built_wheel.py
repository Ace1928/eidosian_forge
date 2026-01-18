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
def _find_already_built_wheel(metadata_directory):
    """Check for a wheel already built during the get_wheel_metadata hook.
    """
    if not metadata_directory:
        return None
    metadata_parent = os.path.dirname(metadata_directory)
    if not os.path.isfile(pjoin(metadata_parent, WHEEL_BUILT_MARKER)):
        return None
    whl_files = glob(os.path.join(metadata_parent, '*.whl'))
    if not whl_files:
        print('Found wheel built marker, but no .whl files')
        return None
    if len(whl_files) > 1:
        print('Found multiple .whl files; unspecified behaviour. Will call build_wheel.')
        return None
    return whl_files[0]