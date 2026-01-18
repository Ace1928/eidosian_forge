from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
def RequireProjectID(args):
    """Prohibit specifying project as a project number.

  Most APIs accept both project number and project id, some of them accept only
  project ids.

  Args:
     args: argparse.namespace, the parsed arguments from the command line
  """
    if args.project:
        if args.project.isdigit():
            raise properties.InvalidValueError("The value of ``--project'' flag was set to Project number.To use this command, set it to PROJECT ID instead.")
        else:
            return
    else:
        proj = properties.VALUES.core.project.Get()
        if proj and proj.isdigit():
            raise properties.InvalidValueError("The value of ``core/project'' property is set to project number.To use this command, set ``--project'' flag to PROJECT ID or set ``core/project'' property to PROJECT ID.")