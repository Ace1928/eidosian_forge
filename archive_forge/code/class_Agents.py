from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Agents(base.Group):
    """Manage Transfer Service agents.

  Manage agents. Agents arre lightweight applications that enable Transfer
  Service users to transfer data to or from POSIX filesystems, such as
  on-premises filesystems. Agents are installed locally on your machine and run
  within Docker containers.
  """