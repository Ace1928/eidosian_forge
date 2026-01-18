from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
@staticmethod
def _DictFromURL(url, command_path, is_extra_repo=False):
    """Loads a json dictionary from a URL.

    Args:
      url: str, The URL to the file to load.
      command_path: the command path to include in the User-Agent header if the
        URL is HTTP
      is_extra_repo: bool, True if this is not the primary repository.

    Returns:
      A ComponentSnapshot object.

    Raises:
      URLFetchError: If the URL cannot be fetched.
    """
    extra_repo = url if is_extra_repo else None
    try:
        response = installers.MakeRequest(url, command_path)
    except requests.exceptions.HTTPError:
        log.debug('Could not fetch [{url}]'.format(url=url), exc_info=True)
        response = None
    if response is None:
        raise URLFetchError(extra_repo=extra_repo)
    code = response.status_code
    if code != requests.codes.ok:
        raise URLFetchError(code=code, extra_repo=extra_repo)
    try:
        data = json.loads(response.content.decode('utf-8'))
        return data
    except ValueError as e:
        log.debug('Failed to parse snapshot [{}]: {}'.format(url, e))
        raise MalformedSnapshotError()