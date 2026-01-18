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
class CommandSearchResults(object):
    """Class to hold the results of a search."""

    def __init__(self, results_data):
        """Create a CommandSearchResults object.

    Args:
      results_data: {str: str}, a dictionary from terms to the locations where
        they were found. Empty string values in the dict represent terms that
        were searched but not found. Locations should be formatted as
        dot-separated strings representing the location in the command (as
        created by LocateTerms above).
    """
        self._results_data = results_data

    def AllTerms(self):
        """Gets a list of all terms that were searched."""
        return self._results_data.keys()

    def FoundTermsMap(self):
        """Gets a map from all terms that were found to their locations."""
        return {k: v for k, v in six.iteritems(self._results_data) if v}