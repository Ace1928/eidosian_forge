import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class SuppressRaggedEOFs(Setting):
    name = 'suppress_ragged_eofs'
    section = 'SSL'
    cli = ['--suppress-ragged-eofs']
    action = 'store_true'
    default = True
    validator = validate_bool
    desc = "    Suppress ragged EOFs (see stdlib ssl module's)\n    "