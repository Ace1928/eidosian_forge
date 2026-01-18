from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetTty(container, command):
    """Determine the ssh command should be run in a TTY or not.

  Args:
    container: str or None, name of container to enter during connection.
    command: [str] or None, the remote command to execute. If no command is
      given, allocate a TTY.

  Returns:
    Bool or None, whether to enforce TTY or not, or None if "auto".
  """
    return True if container and (not command) else None