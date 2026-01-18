import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
def _pkg_tmp():
    tmp = '/tmp/ray/packaging'
    os.makedirs(tmp, exist_ok=True)
    return tmp