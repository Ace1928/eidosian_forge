import glob
import logging
import os
import shutil
import subprocess
import time
from apache_beam.io import gcsio
def _read_local_file_stream(path):
    with open(path) as file_in:
        for line in file_in:
            yield line