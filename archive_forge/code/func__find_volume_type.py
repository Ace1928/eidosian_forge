import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _find_volume_type(cs, volume_type):
    """Get a volume type by ID."""
    return utils.find_resource(cs.volume_types, volume_type)