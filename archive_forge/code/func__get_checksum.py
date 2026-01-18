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
def _get_checksum(name, part=None):
    """Retrieve the checksum of the model/dataset from gensim-data repository.

    Parameters
    ----------
    name : str
        Dataset/model name.
    part : int, optional
        Number of part (for multipart data only).

    Returns
    -------
    str
        Retrieved checksum of dataset/model.

    """
    information = info()
    corpora = information['corpora']
    models = information['models']
    if part is None:
        if name in corpora:
            return information['corpora'][name]['checksum']
        elif name in models:
            return information['models'][name]['checksum']
    elif name in corpora:
        return information['corpora'][name]['checksum-{}'.format(part)]
    elif name in models:
        return information['models'][name]['checksum-{}'.format(part)]