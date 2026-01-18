from __future__ import absolute_import
import argparse
import os
import io
import json
import logging
import sys
import errno
import hashlib
import math
import shutil
import tempfile
from functools import partial
def _get_parts(name):
    """Retrieve the number of parts in which dataset/model has been split.

    Parameters
    ----------
    name: str
        Dataset/model name.

    Returns
    -------
    int
        Number of parts in which dataset/model has been split.

    """
    information = info()
    corpora = information['corpora']
    models = information['models']
    if name in corpora:
        return information['corpora'][name]['parts']
    elif name in models:
        return information['models'][name]['parts']