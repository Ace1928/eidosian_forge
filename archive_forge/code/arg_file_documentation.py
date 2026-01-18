from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
Tab-completion function for ARGSPECs in the format ARG_FILE:ARG_GROUP.

  If the ARG_FILE exists, parse it on-the-fly to get the list of every ARG_GROUP
  it contains. If the ARG_FILE does not exist or the ARGSPEC does not yet
  contain a colon, then fall back to standard shell filename completion by
  returning an empty list.

  Args:
    prefix: the partial ARGSPEC string typed by the user so far.
    parsed_args: the argparse.Namespace for all args parsed so far.
    **kwargs: keyword args, not used.

  Returns:
    The list of all ARG_FILE:ARG_GROUP strings which match the prefix.
  