from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def _get_bundle_version(self, bundle_uri):
    """Get the firmware version from a bundle file, and whether or not it is multi-tenant.

        Only supports HTTP at this time.  Assumes URI exists and is a tarfile.
        Looks for a file oobm-[version].pkg, such as 'oobm-4.0.13.pkg`.  Extracts the version number
        from that filename (in the above example, the version number is "4.0.13".

        To determine if the bundle is multi-tenant or not, it looks inside the .bin file within the tarfile,
        and checks the appropriate byte in the file.

        :param str bundle_uri:  HTTP URI of the firmware bundle.
        :return: Firmware version number contained in the bundle, and whether or not the bundle is multi-tenant.
        Either value will be None if unable to determine.
        :rtype: str or None, bool or None
        """
    bundle_temp_filename = fetch_file(module=self.module, url=bundle_uri)
    if not tarfile.is_tarfile(bundle_temp_filename):
        return (None, None)
    tf = tarfile.open(bundle_temp_filename)
    pattern_pkg = 'oobm-(.+)\\.pkg'
    pattern_bin = '(.*\\.bin)'
    bundle_version = None
    is_multi_tenant = None
    for filename in tf.getnames():
        match_pkg = re.match(pattern_pkg, filename)
        if match_pkg is not None:
            bundle_version = match_pkg.group(1)
        match_bin = re.match(pattern_bin, filename)
        if match_bin is not None:
            bin_filename = match_bin.group(1)
            bin_file = tf.extractfile(bin_filename)
            bin_file.seek(11)
            byte_11 = bin_file.read(1)
            is_multi_tenant = byte_11 == b'\x80'
    return (bundle_version, is_multi_tenant)