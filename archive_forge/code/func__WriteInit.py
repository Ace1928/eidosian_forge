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
def _WriteInit(codegen):
    with util.Chdir(codegen.outdir):
        with io.open('__init__.py', 'w') as out:
            codegen.WriteInit(out)