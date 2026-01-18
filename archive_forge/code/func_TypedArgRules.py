from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.ios import catalog_manager
from googlecloudsdk.calliope import exceptions
def TypedArgRules():
    """Returns the rules for iOS test args which depend on the test type.

  This dict is declared in a function rather than globally to avoid garbage
  collection issues during unit tests.

  Returns:
    A dict keyed by whether type-specific args are required or optional, and
    with a nested dict containing any default values for those shared args.
  """
    return {'xctest': {'required': ['test'], 'optional': ['xcode_version', 'xctestrun_file', 'test_special_entitlements'], 'defaults': {'test_special_entitlements': False}}, 'game-loop': {'required': ['app'], 'optional': ['scenario_numbers'], 'defaults': {'scenario_numbers': [1]}}, 'robo': {'required': ['app'], 'optional': ['test_special_entitlements', 'robo_script'], 'defaults': {'test_special_entitlements': False}}}