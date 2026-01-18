from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import subprocess
import sys
from googlecloudsdk.core.util import files
def ClearPyCache(root_dir=None):
    """Removes generic `__pycache__` folder and  '*.pyc' '*.pyo' files."""
    root_dir = root_dir or files.GetCWD()
    is_cleaned = False
    for name in os.listdir(root_dir):
        item = os.path.join(root_dir, name)
        if os.path.isdir(item):
            if name == '__pycache__':
                files.RmTree(item)
                is_cleaned = True
        else:
            _, ext = os.path.splitext(name)
            if ext in ['.pyc', '.pyo']:
                os.remove(item)
                is_cleaned = True
    return is_cleaned