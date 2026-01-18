import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _find_instance(cs, instance):
    """Get an instance by ID."""
    return utils.find_resource(cs.instances, instance)