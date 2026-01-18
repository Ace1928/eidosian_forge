from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import path_simplifier
def PermittedConnection(permitted_connection):
    return {'interconnectLocation': path_simplifier.Name(permitted_connection.interconnectLocation)}