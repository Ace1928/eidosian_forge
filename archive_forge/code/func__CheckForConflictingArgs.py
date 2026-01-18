from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.android import catalog_manager
from googlecloudsdk.calliope import exceptions
def _CheckForConflictingArgs(self, args):
    """Check for any args that cannot appear simultaneously."""
    if args.device:
        if args.device_ids:
            raise exceptions.ConflictingArgumentsException('--device-ids', '--device')
        if args.os_version_ids:
            raise exceptions.ConflictingArgumentsException('--os-version-ids', '--device')
        if args.locales:
            raise exceptions.ConflictingArgumentsException('--locales', '--device')
        if args.orientations:
            raise exceptions.ConflictingArgumentsException('--orientations', '--device')