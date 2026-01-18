from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import connection_profiles
from googlecloudsdk.core import log
Create a connection profile and wait for its LRO to complete, if necessary.
    