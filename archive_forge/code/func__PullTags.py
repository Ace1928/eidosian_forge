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
def _PullTags(local_repo, client_wrapper, target_dir):
    """Pull tags from 'client_wrapper' into 'local_repo'.

  Args:
    local_repo: (repo.Repo)
    client_wrapper: (ClientWrapper)
    target_dir: (str) The directory of the local repo (for error reporting).

  Returns:
    (str, dulwich.objects.Commit) The tag that was actually pulled (we try to
    get "latest" but fall back to "master") and the commit object
    associated with it.

  Raises:
    errors.HangupException: Hangup during communication to a remote repository.
  """
    for ref, obj_id in six.iteritems(client_wrapper.GetRefs()):
        if ref.startswith('refs/tags/'):
            try:
                local_repo[ref] = obj_id
            except WindowsError as ex:
                raise InvalidTargetDirectoryError('Unable to checkout directory {0}: {1}'.format(target_dir, ex.message))
    revision = None
    tag = None
    for tag in (b'refs/tags/latest', b'refs/heads/master'):
        try:
            log.debug('looking up ref %s', tag)
            revision = local_repo[tag]
            break
        except KeyError:
            log.warning('Unable to checkout branch %s', tag)
    else:
        raise AssertionError('No "refs/heads/master" tag found in repository.')
    return (tag, revision)