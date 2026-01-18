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
def AddIosTestArgs(parser):
    """Register args which are specific to iOS test commands.

  Args:
    parser: An argparse parser used to add arguments that follow a command in
        the CLI.
  """
    parser.add_argument('--type', category=base.COMMONLY_USED_FLAGS, choices=['xctest', 'game-loop', 'robo'], help='The type of iOS test to run.')
    parser.add_argument('--test', category=base.COMMONLY_USED_FLAGS, metavar='XCTEST_ZIP', help='The path to the test package (a zip file containing the iOS app and XCTest files). The given path may be in the local filesystem or in Google Cloud Storage using a URL beginning with `gs://`. Note: any .xctestrun file in this zip file will be ignored if *--xctestrun-file* is specified.')
    parser.add_argument('--xctestrun-file', category=base.COMMONLY_USED_FLAGS, metavar='XCTESTRUN_FILE', help='The path to an .xctestrun file that will override any .xctestrun file contained in the *--test* package. Because the .xctestrun file contains environment variables along with test methods to run and/or ignore, this can be useful for customizing or sharding test suites. The given path may be in the local filesystem or in Google Cloud Storage using a URL beginning with `gs://`.')
    parser.add_argument('--xcode-version', category=base.COMMONLY_USED_FLAGS, help='      The version of Xcode that should be used to run an XCTest. Defaults to the\n      latest Xcode version supported in Firebase Test Lab. This Xcode version\n      must be supported by all iOS versions selected in the test matrix. The\n      list of Xcode versions supported by each version of iOS can be viewed by\n      running `$ {parent_command} versions list`.')
    parser.add_argument('--device', category=base.COMMONLY_USED_FLAGS, type=arg_parsers.ArgDict(min_length=1), action='append', metavar='DIMENSION=VALUE', help="      A list of ``DIMENSION=VALUE'' pairs which specify a target device to test\n      against. This flag may be repeated to specify multiple devices. The device\n      dimensions are: *model*, *version*, *locale*, and *orientation*. If any\n      dimensions are omitted, they will use a default value. The default value,\n      and all possible values, for each dimension can be found with the\n      ``list'' command for that dimension, such as `$ {parent_command} models\n      list`. Omitting this flag entirely will run tests against a single device\n      using defaults for every dimension.\n\n      Examples:\n\n      ```\n      --device model=iphone8plus\n      --device version=11.2\n      --device model=ipadmini4,version=11.2,locale=zh_CN,orientation=landscape\n      ```\n      ")
    parser.add_argument('--results-history-name', help='The history name for your test results (an arbitrary string label; default: the bundle ID for the iOS application). All tests which use the same history name will have their results grouped together in the Firebase console in a time-ordered test history list.')
    parser.add_argument('--app', help='The path to the application archive (.ipa file) for game-loop testing. The path may be in the local filesystem or in Google Cloud Storage using gs:// notation. This flag is only valid when *--type* is *game-loop* or *robo*.')
    parser.add_argument('--test-special-entitlements', action='store_true', default=None, help="      Enables testing special app entitlements. Re-signs an app having special\n      entitlements with a new application-identifier. This currently supports\n      testing Push Notifications (aps-environment) entitlement for up to one\n      app in a project.\n\n      Note: Because this changes the app's identifier, make sure none of the\n      resources in your zip file contain direct references to the test app's\n      bundle id.\n      ")