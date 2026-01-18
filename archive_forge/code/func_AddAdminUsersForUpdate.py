from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddAdminUsersForUpdate(parser):
    """Adds admin user configuration flags for update.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    group = parser.add_group('Admin users', mutex=True)
    AddAdminUsers(group)
    AddClearAdminUsers(group)