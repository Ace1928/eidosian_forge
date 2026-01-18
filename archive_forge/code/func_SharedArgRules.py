from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.ios import catalog_manager
from googlecloudsdk.calliope import exceptions
def SharedArgRules():
    """Returns the rules for iOS test args which are shared by all test types.

  This dict is declared in a function rather than globally to avoid garbage
  collection issues during unit tests.

  Returns:
    A dict keyed by whether shared args are required or optional, and with a
    nested dict containing any default values for those shared args.
  """
    return {'required': ['type'], 'optional': ['additional_ipas', 'async_', 'client_details', 'device', 'directories_to_pull', 'network_profile', 'num_flaky_test_attempts', 'other_files', 'record_video', 'results_bucket', 'results_dir', 'results_history_name', 'timeout'], 'defaults': {'async_': False, 'device': [{}], 'num_flaky_test_attempts': 0, 'record_video': True, 'timeout': 900}}