from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.updater import installers
import requests
from six.moves import StringIO
@classmethod
def FromURL(cls, url, command_path=None):
    """Parses release notes from the given URL using the requests library.

    Any error in downloading or parsing release notes is logged and swallowed
    and None is returned.

    Args:
      url: str, The URL to download and parse.
      command_path: str, The command that is calling this for instrumenting the
        user agent for the download.

    Returns:
      ReleaseNotes, the parsed release notes or None if an error occurred.
    """
    try:
        response = installers.MakeRequest(url, command_path)
        if response is None:
            return None
        code = response.status_code
        if code != requests.codes.ok:
            return None
        return cls(response.text)
    except Exception:
        log.debug('Failed to download [{url}]'.format(url=url), exc_info=True)
    return None