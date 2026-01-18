from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import subprocess
from gslib import exception
Runs external terminal command.

  Args:
    command_and_flags (List[str]): Ordered command and flag strings.

  Returns:
    (stdout (str|None), stderr (str|None)) from running command.

  Raises:
    OSError for any issues running the command.
  