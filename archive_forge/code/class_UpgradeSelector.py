from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container.fleet.scopes.rollout_sequencing import base
from googlecloudsdk.core import log
class UpgradeSelector(arg_parsers.ArgDict):
    """Extends the ArgDict type to properly parse --upgrade-selector argument."""

    def __init__(self):
        super(UpgradeSelector, self).__init__(spec={'name': str, 'version': str}, required_keys=['name', 'version'], max_length=2)