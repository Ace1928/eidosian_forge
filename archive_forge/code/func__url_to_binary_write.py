from __future__ import print_function, absolute_import, division
from . import __version__
from .report import report
import os
import importlib
import inspect
import argparse
import distutils.dir_util
import shutil
from collections import OrderedDict
import glob
import sys
import tarfile
import time
import zipfile
import yaml
def _url_to_binary_write(url, output_path, title):
    """Given a url, output_path and title,
    write the contents of a requests get operation to
    the url in binary mode and print the title of operation"""
    print('Downloading {0}'.format(title))
    resp = requests.get(url, stream=True)
    try:
        with open(output_path, 'wb') as f:
            total_length = int(resp.headers.get('content-length'))
            for chunk in bar(resp.iter_content(chunk_size=1024), expected_size=total_length / 1024 + 1, every=1000):
                if chunk:
                    f.write(chunk)
                    f.flush()
    except:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise