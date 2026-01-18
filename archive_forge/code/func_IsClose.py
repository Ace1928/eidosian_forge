from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def IsClose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Checks if two numerical values are same or almost the same.

  This function is only created to provides backwards compatability for python2
  which does not support 'math.isclose(...)' function. The output of this
  function mimicks exactly the behavior of math.isclose.

  Args:
    a: One of the values to be tested for relative closeness.
    b: One of the values to be tested for relative closeness.
    rel_tol: Relative tolerance allowed. Default value is set so that the two
      values must be equivalent to 9 decimal digits.
    abs_tol: The minimum absoulute tolerance difference. Useful for
      comparisons near zero.

  Returns:
    True if the attribute needs to be updated to the new value, False otherwise.
  """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)