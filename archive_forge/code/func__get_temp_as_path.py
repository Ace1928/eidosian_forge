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
def _get_temp_as_path(account=AS_TEST_ACCOUNT, container=AS_TEST_CONTAINER):
    random_id = ''.join((random.choice(string.ascii_lowercase) for i in range(16)))
    path = f'https://{account}.blob.core.windows.net/{container}/' + random_id
    yield (path + '/file.name')
    sp.run(['az', 'storage', 'blob', 'delete-batch', '--account-name', account, '--source', container, '--pattern', f'{random_id}/*'], check=True, shell=platform.system() == 'Windows')