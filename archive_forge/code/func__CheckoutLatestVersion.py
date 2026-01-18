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
def _CheckoutLatestVersion(target_dir, url):
    """Pull tags and checkout the latest version of the target directory.

  Args:
    target_dir: (str) Directory name.
    url: (str) Git repository URL.

  Raises:
    errors.HangupException: Hangup during communication to a remote repository.
  """
    local_repo = repo.Repo(target_dir)
    try:
        client_wrapper = WrapClient(url)
        local_repo = repo.Repo(target_dir)
        tag, revision = _PullTags(local_repo, client_wrapper, target_dir)
        log.info('Checking out revision [%s] of [%s] into [%s]', tag, url, target_dir)
        try:
            index.build_index_from_tree(local_repo.path, local_repo.index_path(), local_repo.object_store, revision.tree)
        except (IOError, OSError, WindowsError) as ex:
            raise InvalidTargetDirectoryError('Unable to checkout directory {0}: {1}'.format(target_dir, ex.message))
    except AssertionError as ex:
        raise InvalidRepositoryError()
    finally:
        local_repo.close()