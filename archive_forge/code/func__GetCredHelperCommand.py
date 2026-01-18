from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import subprocess
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
import uritemplate
def _GetCredHelperCommand(uri, full_path=False, min_version=_HELPER_MIN):
    """Returns the gcloud credential helper command for a remote repository.

  The command will be of the form '!gcloud auth git-helper --account=EMAIL
  --ignore-unknown $@`. See https://git-scm.com/docs/git-config. If the
  installed version of git or the remote repository does not support
  the gcloud credential helper, then returns None.

  Args:
    uri: str, The uri of the remote repository.
    full_path: bool, If true, use the full path to gcloud.
    min_version: minimum git version; if found git is earlier than this, warn
        and return None

  Returns:
    str, The credential helper command if it is available.
  """
    credentialed_hosts = ['source.developers.google.com']
    extra = properties.VALUES.core.credentialed_hosted_repo_domains.Get()
    if extra:
        credentialed_hosts.extend(extra.split(','))
    if any((uri.startswith('https://' + host + '/') for host in credentialed_hosts)):
        try:
            CheckGitVersion(min_version)
        except GitVersionException as e:
            helper_min_str = '.'.join((six.text_type(i) for i in min_version))
            log.warning(textwrap.dedent('          You are using a Google-hosted repository with a\n          {current} which is older than {min_version}. If you upgrade\n          to {min_version} or later, gcloud can handle authentication to\n          this repository. Otherwise, to authenticate, use your Google\n          account and the password found by running the following command.\n           $ gcloud auth print-access-token'.format(current=e.cur_version, min_version=helper_min_str)))
            return None
        return '!{0} auth git-helper --account={1} --ignore-unknown $@'.format(_GetGcloudScript(full_path), properties.VALUES.core.account.Get(required=True))
    return None