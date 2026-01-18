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
def _filesystem_uri(path):
    if os.name == 'nt':
        uri = 'file:///{}'.format(path)
    else:
        uri = 'file://{}'.format(path)
    return uri