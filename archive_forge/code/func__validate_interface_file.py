import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
def _validate_interface_file(interface_file: str):
    if not os.path.exists(interface_file):
        raise ValueError('Interface file does not exist: {}'.format(interface_file))
    for line in open(interface_file):
        line = line.replace('\n', '')
        if line.startswith('import ') or line.startswith('from '):
            if line != 'import ray' and 'noqa' not in line:
                raise ValueError('Interface files are only allowed to import `ray` at top-level, found `{}`. Please either remove or change this into a lazy import. To unsafely allow this import, add `# noqa` to the line in question.'.format(line))