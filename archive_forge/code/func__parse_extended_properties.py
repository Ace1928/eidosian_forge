import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _parse_extended_properties(extended_properties):
    return dict([(k, v) for k, v in [kv.strip().split('=') for kv in extended_properties.split(',')]])