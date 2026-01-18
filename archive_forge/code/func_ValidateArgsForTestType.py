from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def ValidateArgsForTestType(args, test_type, type_rules, shared_rules, all_test_args_set):
    """Raise errors if required args are missing or invalid args are present.

  Args:
    args: an argparse.Namespace object which contains attributes for all the
      arguments that were provided to the command invocation (i.e. command
      group and command arguments combined).
    test_type: string containing the type of test to run.
    type_rules: a nested dictionary defining the required and optional args
      per type of test, plus any default values.
    shared_rules: a nested dictionary defining the required and optional args
      shared among all test types, plus any default values.
    all_test_args_set: a set of strings for every gcloud-test argument to use
      for validation.

  Raises:
    InvalidArgException: If an arg doesn't pair with the test type.
    RequiredArgumentException: If a required arg for the test type is missing.
  """
    required_args = type_rules[test_type]['required'] + shared_rules['required']
    optional_args = type_rules[test_type]['optional'] + shared_rules['optional']
    allowable_args_for_type = required_args + optional_args
    for arg in all_test_args_set:
        if getattr(args, arg, None) is not None:
            if arg not in allowable_args_for_type:
                raise test_exceptions.InvalidArgException(arg, 'may not be used with test type [{0}].'.format(test_type))
    for arg in required_args:
        if getattr(args, arg, None) is None:
            raise exceptions.RequiredArgumentException('{0}'.format(test_exceptions.ExternalArgNameFrom(arg)), 'must be specified with test type [{0}].'.format(test_type))