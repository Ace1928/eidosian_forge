from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateAppEngineVersionsURIs(unused_ref, args, request):
    """Checks if all provided App Engine version URIs are in correct format."""
    flags = ['source_app_engine_version']
    version_pattern = re.compile('apps/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/services/[-\\w]+/versions/[-\\w]+')
    for flag in flags:
        if not args.IsSpecified(flag):
            continue
        version = getattr(args, flag)
        if not version_pattern.match(version):
            raise InvalidInputError('Invalid value for flag {}: {}\nExpected App Engine version in the following format:\n  apps/my-project/services/my-service/versions/my-version'.format(flag, version))
    return request