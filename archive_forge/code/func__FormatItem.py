from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def _FormatItem(item):
    """Helper function to reformat string as markdown list item: {STRING}::."""
    return '{}::'.format(item)