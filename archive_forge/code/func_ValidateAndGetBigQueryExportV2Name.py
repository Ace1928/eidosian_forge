from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def ValidateAndGetBigQueryExportV2Name(args):
    """Returns relative resource name for a v2 Big Query export.

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
  """
    id_pattern = re.compile('^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')
    nonregionalized_resource_pattern = re.compile('(organizations|projects|folders)/.+/bigQueryExports/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')
    regionalized_resource_pattern = re.compile('(organizations|projects|folders)/.+/locations/.+/bigQueryExports/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')
    bq_export_id = args.BIG_QUERY_EXPORT
    location = util.ValidateAndGetLocation(args, 'v2')
    if id_pattern.match(bq_export_id):
        parent = util.GetParentFromNamedArguments(args)
        if parent is None:
            raise errors.InvalidSCCInputError('BigQuery export must match the full resource name, or `--organization=`, `--folder=` or `--project=` must be provided.')
        return f'{parent}/locations/{location}/bigQueryExports/{bq_export_id}'
    if regionalized_resource_pattern.match(bq_export_id):
        return bq_export_id
    if nonregionalized_resource_pattern.match(bq_export_id):
        [parent_segment, id_segment] = bq_export_id.split('/bigQueryExports/')
        return f'{parent_segment}/locations/{location}/bigQueryExports/{id_segment}'
    raise errors.InvalidSCCInputError('BigQuery export must match (organizations|projects|folders)/.+/bigQueryExports/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ (organizations|projects|folders)/.+/locations/.+/bigQueryExports/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ or [a-zA-Z0-9-_]{1,128}$.')