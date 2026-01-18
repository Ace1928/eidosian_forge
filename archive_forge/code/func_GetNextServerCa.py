from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetNextServerCa(sql_client, sql_messages, instance_ref):
    """Returns the upcoming Server CA Cert."""
    server_ca_types = GetServerCaTypeDict(ListServerCas(sql_client, sql_messages, instance_ref))
    return server_ca_types.get(NEXT_CERT_LABEL)