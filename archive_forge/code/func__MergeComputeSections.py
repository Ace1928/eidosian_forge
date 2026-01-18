from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
import stat
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _MergeComputeSections(existing_content, compute_section):
    """Merges a new compute section into an existing compute section.

  Appends new host entries into the compute section. If any of the newly
  appended entries match the host of a previous compute section entry, the
  previous entry is removed. The resulting merged configuration file contents
  are returned. This function has no side effects, and does not directly change
  the contents of the configuration file.

  Args:
   existing_content: str, Current content of ssh config file.
   compute_section: str, New content to be added to the compute section.

  Raises:
    MultipleComputeSectionsError: If the config file has multiple compute
      sections already.

  Returns:
    A string containing the new contents of the ssh configuration file after
    merging the new entries for the compute section.
  """
    match = _SECTION_RE.search(existing_content)
    if not match:
        if compute_section:
            if existing_content[-1] != '\n':
                existing_content += '\n'
            if existing_content[-2:] != '\n\n':
                existing_content += '\n'
            new_content = existing_content + compute_section
        else:
            new_content = existing_content
    elif not _ComputeSectionValid(existing_content):
        raise MultipleComputeSectionsError()
    elif not compute_section:
        return existing_content
    else:
        new_hosts = _HOST_RE.findall(compute_section)
        existing_hosts = match.group(2).split('\n\n')
        hosts = [host for host in existing_hosts if _HOST_RE.search(host).group(1) not in new_hosts]
        hosts.append(_SECTION_RE.search(compute_section).group(2))
        compute_section = '\n\n'.join(hosts)
        new_content = existing_content[0:match.start(2)] + compute_section + existing_content[match.end(2):]
    return new_content