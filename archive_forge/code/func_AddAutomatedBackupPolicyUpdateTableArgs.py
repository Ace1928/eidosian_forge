from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def AddAutomatedBackupPolicyUpdateTableArgs():
    """Adds automated backup policy commands to update table CLI."""
    return [base.Argument('--enable-automated-backup', help='Once set, enables the default automated backup policy (retention_period=72h, frequency=24h) for the table.', action='store_true'), base.Argument('--disable-automated-backup', help='Once set, disables automated backup policy for the table.', action='store_true')]