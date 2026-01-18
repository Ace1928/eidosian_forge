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
class ReleaseNotes(object):
    """Represents a parsed RELEASE_NOTES file.

  The file should have the general structure of:

  # Google Cloud SDK - Release Notes

  Copyright 2014-2015 Google LLC. All rights reserved.

  ## 0.9.78 (2015/09/16)

  *   Note
  *   Note 2

  ## 0.9.77 (2015/09/09)

  *   Note 3
  """
    _VERSION_SPLIT_REGEX = '(?<=\\n)\\n## +(?P<version>\\S+).*\\n(?:\\n.*(?!\\n\\n## ))+.'
    MAX_DIFF = 15

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

    def __init__(self, text):
        """Parse the release notes from the given text.

    Args:
      text: str, The text of the release notes to parse.

    Returns:
      ReleaseNotes, the parsed release notes.
    """
        self._text = text.replace('\r\n', '\n')
        versions = []
        for m in re.finditer(ReleaseNotes._VERSION_SPLIT_REGEX, self._text):
            versions.append((m.group('version'), m.group().strip()))
        self._versions = versions

    def GetVersionText(self, version):
        """Gets the release notes text for the given version.

    Args:
      version: str, The version to get the release notes for.

    Returns:
      str, The release notes or None if the version does not exist.
    """
        index = self._GetVersionIndex(version)
        if index is None:
            return None
        return self._versions[index][1]

    def _GetVersionIndex(self, version):
        """Gets the index of the given version in the list of parsed versions.

    Args:
      version: str, The version to get the index for.

    Returns:
      int, The index of the given version or None if not found.
    """
        for i, (v, _) in enumerate(self._versions):
            if v == version:
                return i
        return None

    def Diff(self, start_version, end_version):
        """Creates a diff of the release notes between the two versions.

    The release notes are returned in reversed order (most recent first).

    Args:
      start_version: str, The version at which to start the diff.  This should
        be the later of the two versions.  The diff will start with this version
        and go backwards in time until end_version is hit.  If None, the diff
        will start at the most recent entry.
      end_version: str, The version at which to stop the diff.  This should be
        the version you are currently on.  The diff is accumulated until this
        version it hit.  This version is not included in the diff.  If None,
        the diff will include through the end of all release notes.

    Returns:
      [(version, text)], The list of release notes in the diff from most recent
      to least recent.  Each item is a tuple of the version string and the
      release notes text for that version.  Returns None if either of the
      versions are not present in the release notes.
    """
        if start_version:
            start_index = self._GetVersionIndex(start_version)
            if start_index is None:
                return None
        else:
            start_index = 0
        if end_version:
            end_index = self._GetVersionIndex(end_version)
            if end_index is None:
                return None
        else:
            end_index = len(self._versions)
        return self._versions[start_index:end_index]