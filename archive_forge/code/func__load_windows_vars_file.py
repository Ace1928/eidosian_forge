from __future__ import absolute_import, division, print_function
import argparse
import gzip
import pathlib
import shutil
import subprocess
import sys
from urllib import request
from xml.etree import ElementTree
import yaml
def _load_windows_vars_file(filename):
    with open(filename, 'r') as fd:
        return yaml.safe_load(fd)