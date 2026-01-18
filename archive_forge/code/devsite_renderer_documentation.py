from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.document_renderers import devsite_scripts
from googlecloudsdk.core.document_renderers import html_renderer
Add global flags links to line if any.

    Args:
      line: The text line.

    Returns:
      line with annoted global flag links.
    