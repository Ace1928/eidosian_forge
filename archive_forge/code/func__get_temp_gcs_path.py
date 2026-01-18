import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
@contextlib.contextmanager
def _get_temp_gcs_path():
    path = f'gs://{GCS_TEST_BUCKET}/' + ''.join((random.choice(string.ascii_lowercase) for i in range(16)))
    gfile.mkdir(path)
    yield (path + '/file.name')
    gfile.rmtree(path)