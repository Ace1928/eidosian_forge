import argparse
import functools
import hashlib
import logging
import os
from oslo_utils import encodeutils
from oslo_utils import importutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
def add_boolean_argument(parser, name, **kwargs):
    for keyword in ('metavar', 'choices'):
        kwargs.pop(keyword, None)
    default = kwargs.pop('default', argparse.SUPPRESS)
    parser.add_argument(name, metavar='{True,False}', choices=['True', 'true', 'False', 'false'], default=default, **kwargs)