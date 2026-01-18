from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
from dulwich import client
from dulwich import errors
from dulwich import index
from dulwich import porcelain
from dulwich import repo
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
def InstallRuntimeDef(url, target_dir):
    """Install a runtime definition in the target directory.

  This installs the runtime definition identified by 'url' into the target
  directory.  At this time, the runtime definition url must be the URL of a
  git repository and we identify the tree to checkout based on 1) the presence
  of a "latest" tag ("refs/tags/latest") 2) if there is no "latest" tag, the
  head of the "master" branch ("refs/heads/master").

  Args:
    url: (str) A URL identifying a git repository.  The HTTP, TCP and local
      git protocols are formally supported. e.g.
        https://github.com/project/repo.git
        git://example.com:1234
        /some/git/directory
    target_dir: (str) The path where the definition will be installed.

  Raises:
    InvalidTargetDirectoryError: An invalid target directory was specified.
    RepositoryCommunicationError: An error occurred communicating to the
      repository identified by 'url'.
  """
    try:
        _FetchRepo(target_dir, url)
        _CheckoutLatestVersion(target_dir, url)
    except (errors.HangupException, errors.GitProtocolError) as ex:
        raise RepositoryCommunicationError('An error occurred accessing {0}: {1}'.format(url, ex.message))