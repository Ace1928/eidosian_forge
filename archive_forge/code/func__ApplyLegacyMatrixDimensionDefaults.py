from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.android import catalog_manager
from googlecloudsdk.calliope import exceptions
def _ApplyLegacyMatrixDimensionDefaults(self, args):
    """Apply defaults to each dimension flag only if not using sparse matrix."""
    if args.device:
        return
    if not (args.device_ids or args.os_version_ids or args.locales or args.orientations):
        args.device = [{}]
        return
    if not args.device_ids:
        args.device_ids = [self._catalog_mgr.GetDefaultModel()]
    if not args.os_version_ids:
        args.os_version_ids = [self._catalog_mgr.GetDefaultVersion()]
    if not args.locales:
        args.locales = [self._catalog_mgr.GetDefaultLocale()]
    if not args.orientations:
        args.orientations = [self._catalog_mgr.GetDefaultOrientation()]