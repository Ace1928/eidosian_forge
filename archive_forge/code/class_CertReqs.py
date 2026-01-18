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
class CertReqs(Setting):
    name = 'cert_reqs'
    section = 'SSL'
    cli = ['--cert-reqs']
    validator = validate_pos_int
    default = ssl.CERT_NONE
    desc = "    Whether client certificate is required (see stdlib ssl module's)\n\n    ===========  ===========================\n    --cert-reqs      Description\n    ===========  ===========================\n    `0`          no client veirifcation\n    `1`          ssl.CERT_OPTIONAL\n    `2`          ssl.CERT_REQUIRED\n    ===========  ===========================\n    "