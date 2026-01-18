from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
Returns relative resource name for a v2 Big Query export.

  Validates on regexes for args containing full names with locations or short
  names with resources.

  Args:
    args: an argparse object that should contain .BIG_QUERY_EXPORT, optionally 1
      of .organization, .folder, .project; and optionally .location

  Examples:

  args with BIG_QUERY_EXPORT="organizations/123/bigQueryExports/config1"
  and location="locations/us" returns
  organizations/123/locations/us/bigQueryExports/config1

  args with
  BIG_QUERY_EXPORT="folders/123/locations/us/bigQueryExports/config1"
  and returns folders/123/locations/us/bigQueryExports/config1

  args with BIG_QUERY_EXPORT="config1", projects="projects/123", and
  locations="us" returns projects/123/bigQueryExports/config1
  