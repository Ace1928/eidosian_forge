from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.command_lib.logs import stream
import six
def _SplitMultiline(log_generator, allow_multiline=False):
    """Splits the dict output of logs into multiple lines.

  Args:
    log_generator: iterator that returns a an ml log in dict format.
    allow_multiline: Tells us if logs with multiline messages are okay or not.

  Yields:
    Single-line ml log dictionaries.
  """
    for log in log_generator:
        log_dict = _EntryToDict(log)
        messages = log_dict['message'].splitlines()
        if allow_multiline:
            yield log_dict
        else:
            if not messages:
                messages = ['']
            for message in messages:
                single_line_log = copy.deepcopy(log_dict)
                single_line_log['message'] = message
                yield single_line_log