import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def __make_cutensor_record(cuda_version, public_version, filename_linux, filename_windows):
    return {'cuda': cuda_version, 'cutensor': public_version, 'assets': {'Linux': {'url': _make_cutensor_url('linux', filename_linux), 'filenames': ['libcutensor.so.{}'.format(public_version)]}, 'Windows': {'url': _make_cutensor_url('windows', filename_windows), 'filenames': ['cutensor.dll']}}}