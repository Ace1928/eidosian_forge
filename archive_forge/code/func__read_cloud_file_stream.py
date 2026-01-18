import glob
import logging
import os
import shutil
import subprocess
import time
from apache_beam.io import gcsio
def _read_cloud_file_stream(path):
    read_file = subprocess.Popen(['gsutil', 'cp', path, '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with read_file.stdout as file_in:
        for line in file_in:
            yield line
    if read_file.wait() != 0:
        raise IOError('Unable to read data from Google Cloud Storage: "%s"' % path)