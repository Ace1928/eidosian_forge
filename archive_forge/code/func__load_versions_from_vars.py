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
def _load_versions_from_vars(vars):
    return set((tuple((int(c) for c in item['version'].split('.'))) + (item['build'],) for item in vars['_msi_lookup'].values()))