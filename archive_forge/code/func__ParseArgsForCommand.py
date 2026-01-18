from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.ondemandscanning import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import ondemandscanning_util as ods_util
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import platforms
import six
def _ParseArgsForCommand(self, resource_uri, remote, fake_extraction, additional_package_types, experimental_package_types, verbose_errors, **kwargs):
    args = ['--resource_uri=' + resource_uri, '--remote=' + six.text_type(remote), '--provide_fake_results=' + six.text_type(fake_extraction), '--undefok=' + ','.join(['additional_package_types', 'verbose_errors'])]
    package_types = []
    if additional_package_types:
        package_types += additional_package_types
    if experimental_package_types:
        package_types += experimental_package_types
    if package_types:
        args.append('--additional_package_types=' + six.text_type(','.join(package_types)))
    if verbose_errors:
        args.append('--verbose_errors=' + six.text_type(verbose_errors))
    return args