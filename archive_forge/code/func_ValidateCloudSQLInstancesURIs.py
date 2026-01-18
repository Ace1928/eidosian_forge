from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateCloudSQLInstancesURIs(unused_ref, args, request):
    """Checks if all provided Cloud SQL Instances URIs are in correct format."""
    flags = ['source_cloud_sql_instance', 'destination_cloud_sql_instance']
    instance_pattern = re.compile('projects/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/instances/[-\\w]+')
    for flag in flags:
        if args.IsSpecified(flag):
            instance = getattr(args, flag)
            if not instance_pattern.match(instance):
                raise InvalidInputError('Invalid value for flag {}: {}\nExpected Cloud SQL instance in the following format:\n  projects/my-project/instances/my-instance'.format(flag, instance))
    return request