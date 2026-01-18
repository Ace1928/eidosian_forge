from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.containeranalysis import util as containeranalysis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.images.packages import exceptions
from googlecloudsdk.command_lib.compute.images.packages import filter_utils
from googlecloudsdk.command_lib.compute.images.packages import flags as package_flags
def _GetDiff(self, args, package_versions_base, package_versions_diff):
    all_package_names = set(package_versions_base.keys()).union(set(package_versions_diff.keys()))
    show_all_diff_packages = True
    if args.show_added_packages or args.show_removed_packages or args.show_updated_packages:
        show_all_diff_packages = False
    diff = []
    empty = ('-', '-')
    for package_name in all_package_names:
        versions_base = package_versions_base.get(package_name, [])
        versions_diff = package_versions_diff.get(package_name, [])
        if set(versions_base) != set(versions_diff):
            len_base = len(versions_base)
            len_diff = len(versions_diff)
            if show_all_diff_packages or (args.show_added_packages and len_base == 0 and (len_diff != 0)) or (args.show_removed_packages and len_base != 0 and (len_diff == 0)) or (args.show_updated_packages and len_base != 0 and (len_diff != 0)):
                for idx in range(max(len_base, len_diff)):
                    version_base, revision_base = versions_base[idx] if idx < len_base else empty
                    version_diff, revision_diff = versions_diff[idx] if idx < len_diff else empty
                    package_diff = {'name': package_name, 'version_base': version_base, 'revision_base': revision_base, 'version_diff': version_diff, 'revision_diff': revision_diff}
                    diff.append(package_diff)
    return sorted(diff, key=lambda package_diff: package_diff['name'])