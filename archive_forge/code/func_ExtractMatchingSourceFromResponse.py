from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
def ExtractMatchingSourceFromResponse(response, args):
    """Returns source that matches the user provided source_id or display_name.

  Args:
    response: Response object.
    args: Input arguments.

  Raises:
    Error if it's an invalid source or no matching source was found.
  """
    for source in response:
        if args.source and source.name.endswith(args.source) or (args.source_display_name and source.displayName == args.source_display_name):
            return source
    raise core_exceptions.Error('Source: %s not found.' % (args.source if args.source is not None else args.source_display_name))