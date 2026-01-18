from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis import apis_map
def _GenerateMtlsDisallowedServices():
    """Generates a table for services which do NOT support client certificate."""
    disallowlist = []
    for service, versions in apis_map.MAP.items():
        for version, api_def in versions.items():
            if not api_def.enable_mtls:
                disallowlist.append((service, version))
    disallowlist.sort()
    table_out = io.StringIO()
    table_out.write('\nSERVICE | VERSION | NOTES\n --- | --- | ---\n --- | --- | ---\n')
    for service, version in disallowlist:
        table_out.write('{} | {} |\n'.format(service, version))
    table_out.write('--- | --- | ---\n')
    return table_out.getvalue()