from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def _remove_metadata_headers(headers_dict):
    """Filters out some headers that correspond to metadata fields.

  It's not necessarily important that all headers corresponding to metadata
  fields are filtered here, but failing to do so for some (e.g. content-type)
  can lead to bugs if the user's setting overrides values set by our API
  client that are required for it to function properly.

  Args:
    headers_dict (dict): Header key:value pairs provided by the user.

  Returns:
    A dictionary with a subset of the pairs in headers_dict -- those matching
    some metadata fields are filtered out.
  """
    filtered_headers = {}
    for header, value in headers_dict.items():
        header_matches_metadata_prefixes = (header.startswith(prefix) for prefix in _METADATA_HEADER_PREFIXES)
        if header not in _METADATA_HEADERS and (not any(header_matches_metadata_prefixes)):
            filtered_headers[header] = value
    return filtered_headers