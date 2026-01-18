from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateCloudRunRevisionsURIs(unused_ref, args, request):
    """Checks if all provided Cloud Run revisions URIs are in correct format."""
    flags = ['source_cloud_run_revision']
    revision_pattern = re.compile('projects/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/locations/[-\\w]+/revisions/[-\\w]+')
    for flag in flags:
        if not args.IsSpecified(flag):
            continue
        revision = getattr(args, flag)
        if not revision_pattern.match(revision):
            raise InvalidInputError('Invalid value for flag {}: {}\nExpected Cloud Run revision in the following format:\n  projects/my-project/locations/location/revisions/my-revision'.format(flag, revision))
    return request