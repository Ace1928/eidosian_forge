import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def _ensure_minio_component_version(component, minimum_year):
    full_args = [component, '--version']
    with subprocess.Popen(full_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8') as proc:
        if proc.wait(10) != 0:
            return False
        stdout = proc.stdout.read()
        pattern = component + ' version RELEASE\\.(\\d+)-.*'
        version_match = re.search(pattern, stdout)
        if version_match:
            version_year = version_match.group(1)
            return int(version_year) >= minimum_year
        else:
            raise FileNotFoundError('minio component older than the minimum year')