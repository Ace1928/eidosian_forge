import glob
import logging
import os
import shutil
import subprocess
import time
from apache_beam.io import gcsio
def _read_local_file(local_path):
    with open(local_path, 'r') as f:
        return f.read()