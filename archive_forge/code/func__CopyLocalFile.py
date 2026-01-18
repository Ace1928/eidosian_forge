import argparse
import contextlib
import io
import json
import logging
import os
import pkgutil
import sys
from apitools.base.py import exceptions
from apitools.gen import gen_client_lib
from apitools.gen import util
def _CopyLocalFile(filename):
    with contextlib.closing(io.open(filename, 'w')) as out:
        src_data = pkgutil.get_data('apitools.base.py', filename)
        if src_data is None:
            raise exceptions.GeneratedClientError('Could not find file %s' % filename)
        out.write(src_data)