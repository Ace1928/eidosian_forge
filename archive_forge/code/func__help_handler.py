import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def _help_handler(self, args, screen_info=None):
    """Command handler for "help".

    "help" is a common command that merits built-in support from this class.

    Args:
      args: Command line arguments to "help" (not including "help" itself).
      screen_info: (dict) Information regarding the screen, e.g., the screen
        width in characters: {"cols": 80}

    Returns:
      (RichTextLines) Screen text output.
    """
    _ = screen_info
    if not args:
        return self.get_help()
    elif len(args) == 1:
        return self.get_help(args[0])
    else:
        return RichTextLines(['ERROR: help takes only 0 or 1 input argument.'])