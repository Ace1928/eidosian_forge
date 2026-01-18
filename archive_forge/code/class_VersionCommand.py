from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import sys
import six
import boto
import crcmod
import gslib
from gslib.command import Command
from gslib.utils import system_util
from gslib.utils.boto_util import GetFriendlyConfigFilePaths
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import UTF8
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
class VersionCommand(Command):
    """Implementation of gsutil version command."""
    command_spec = Command.CreateCommandSpec('version', command_name_aliases=['ver'], usage_synopsis=_SYNOPSIS, min_args=0, max_args=0, supported_sub_args='l', file_url_ok=False, provider_url_ok=False, urls_start_arg=0)
    help_spec = Command.HelpSpec(help_name='version', help_name_aliases=['ver'], help_type='command_help', help_one_line_summary='Print version info about gsutil', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})

    def RunCommand(self):
        """Command entry point for the version command."""
        long_form = False
        if self.sub_opts:
            for o, _ in self.sub_opts:
                if o == '-l':
                    long_form = True
        config_paths = ', '.join(GetFriendlyConfigFilePaths())
        shipped_checksum = gslib.CHECKSUM
        try:
            cur_checksum = self._ComputeCodeChecksum()
        except IOError:
            cur_checksum = 'MISSING FILES'
        if shipped_checksum == cur_checksum:
            checksum_ok_str = 'OK'
        else:
            checksum_ok_str = '!= %s' % shipped_checksum
        sys.stdout.write('gsutil version: %s\n' % gslib.VERSION)
        if long_form:
            long_form_output = 'checksum: {checksum} ({checksum_ok})\nboto version: {boto_version}\npython version: {python_version}\nOS: {os_version}\nmultiprocessing available: {multiprocessing_available}\nusing cloud sdk: {cloud_sdk}\npass cloud sdk credentials to gsutil: {cloud_sdk_credentials}\nconfig path(s): {config_paths}\ngsutil path: {gsutil_path}\ncompiled crcmod: {compiled_crcmod}\ninstalled via package manager: {is_package_install}\neditable install: {is_editable_install}\nshim enabled: {is_shim_enabled}\n'
            sys.stdout.write(long_form_output.format(checksum=cur_checksum, checksum_ok=checksum_ok_str, boto_version=boto.__version__, python_version=sys.version.replace('\n', ''), os_version='%s %s' % (platform.system(), platform.release()), multiprocessing_available=CheckMultiprocessingAvailableAndInit().is_available, cloud_sdk=system_util.InvokedViaCloudSdk(), cloud_sdk_credentials=system_util.CloudSdkCredPassingEnabled(), config_paths=config_paths, gsutil_path=GetCloudSdkGsutilWrapperScriptPath() or gslib.GSUTIL_PATH, compiled_crcmod=UsingCrcmodExtension(), is_package_install=gslib.IS_PACKAGE_INSTALL, is_editable_install=gslib.IS_EDITABLE_INSTALL, is_shim_enabled=boto.config.getbool('GSUtil', 'use_gcloud_storage', False)))
        return 0

    def _ComputeCodeChecksum(self):
        """Computes a checksum of gsutil code.

    This checksum can be used to determine if users locally modified
    gsutil when requesting support. (It's fine for users to make local mods,
    but when users ask for support we ask them to run a stock version of
    gsutil so we can reduce possible variables.)

    Returns:
      MD5 checksum of gsutil code.
    """
        if gslib.IS_PACKAGE_INSTALL:
            return 'PACKAGED_GSUTIL_INSTALLS_DO_NOT_HAVE_CHECKSUMS'
        m = GetMd5()
        files_to_checksum = [gslib.GSUTIL_PATH]
        for root, _, files in os.walk(gslib.GSLIB_DIR):
            for filepath in files:
                if filepath.endswith('.py'):
                    files_to_checksum.append(os.path.join(root, filepath))
        for filepath in sorted(files_to_checksum):
            if six.PY2:
                f = open(filepath, 'rb')
                content = f.read()
                content = re.sub('(\\r\\n|\\r|\\n)', b'\n', content)
                m.update(content)
                f.close()
            else:
                f = open(filepath, 'r', encoding=UTF8)
                content = f.read()
                content = re.sub('(\\r\\n|\\r|\\n)', '\n', content)
                m.update(content.encode(UTF8))
                f.close()
        return m.hexdigest()