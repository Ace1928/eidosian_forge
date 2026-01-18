from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.declarative import flags as declarative_config_flags
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
def RunExport(args, collection, resource_ref):
    client = kcc_client.KccClient()
    if getattr(args, 'all', None):
        return client.ExportAll(args=args, collection=collection)
    return client.Export(args, resource_uri=resource_ref)