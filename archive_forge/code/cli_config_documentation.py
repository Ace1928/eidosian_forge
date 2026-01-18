import collections
import json
import os
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile
Get a text summary of the config.

    Args:
      highlight: A property name to highlight in the output.

    Returns:
      A `RichTextLines` output.
    