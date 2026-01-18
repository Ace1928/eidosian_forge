import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
List manageable snapshots.

    Supported by --os-volume-api-version 3.8 or greater.
    