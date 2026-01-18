from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
def GetSetOfAllTestArgs(type_rules, shared_rules):
    """Build a set of all possible 'gcloud test run' args.

  We need this set to test for invalid arg combinations because gcloud core
  adds many args to our args.Namespace that we don't care about and don't want
  to validate. We also need this to validate args coming from an arg-file.

  Args:
    type_rules: a nested dictionary defining the required and optional args
      per type of test, plus any default values.
    shared_rules: a nested dictionary defining the required and optional args
      shared among all test types, plus any default values.

  Returns:
    A set of strings for every gcloud-test argument.
  """
    all_test_args_list = shared_rules['required'] + shared_rules['optional'] + list(shared_rules['defaults'].keys())
    for type_dict in type_rules.values():
        all_test_args_list += type_dict['required'] + type_dict['optional'] + list(type_dict['defaults'].keys())
    return set(all_test_args_list)