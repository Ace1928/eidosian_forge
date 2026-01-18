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
def _fetch_available_versions():
    available_versions = set()
    response = request.urlopen(BASE_REPO_URL + 'repodata/repomd.xml', timeout=30)
    root = ElementTree.parse(response).getroot()
    for data in root.iter('{http://linux.duke.edu/metadata/repo}data'):
        if data.get('type') == 'primary':
            break
    else:
        return available_versions
    location = next(data.iter('{http://linux.duke.edu/metadata/repo}location'))
    path = location.attrib['href']
    response = request.urlopen(BASE_REPO_URL + path, timeout=30)
    root = ElementTree.fromstring(gzip.decompress(response.read()))
    for package in root.iter('{http://linux.duke.edu/metadata/common}package'):
        name = next(package.iter('{http://linux.duke.edu/metadata/common}name'))
        if name.text != 'sensu-go-agent':
            continue
        version = next(package.iter('{http://linux.duke.edu/metadata/common}version'))
        version_tuple = tuple((int(c) for c in version.get('ver').split('.')))
        if version_tuple < MINIMAL_VERSION:
            continue
        available_versions.add(version_tuple + (int(version.get('rel')),))
    return available_versions