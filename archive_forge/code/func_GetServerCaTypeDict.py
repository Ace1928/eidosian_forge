from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetServerCaTypeDict(list_server_cas_response):
    """Gets a dictionary mapping Server CA Cert types to certs.

  The keys to the dictionary returned will be some combinatiaon of 'Current',
  'Next', and 'Previous'.

  Args:
    list_server_cas_response: InstancesListServerCasResponse instance.

  Returns:
    A dictionary mapping Server CA Cert types to SslCert instances.
  """
    server_ca_types = {}
    active_id = list_server_cas_response.activeVersion
    certs = list_server_cas_response.certs
    active_cert = None
    for cert in certs:
        if cert.sha1Fingerprint == active_id:
            active_cert = cert
            break
    if not active_cert:
        return server_ca_types
    server_ca_types[ACTIVE_CERT_LABEL] = active_cert
    inactive_certs = [cert for cert in certs if cert.sha1Fingerprint != active_id]
    if len(inactive_certs) == 1:
        inactive_cert = inactive_certs[0]
        if inactive_cert.createTime > active_cert.createTime:
            server_ca_types[NEXT_CERT_LABEL] = inactive_cert
        else:
            server_ca_types[PREVIOUS_CERT_LABEL] = inactive_cert
    elif len(inactive_certs) > 1:
        inactive_certs = sorted(inactive_certs, key=lambda cert: cert.createTime)
        server_ca_types[PREVIOUS_CERT_LABEL] = inactive_certs[0]
        server_ca_types[NEXT_CERT_LABEL] = inactive_certs[-1]
    return server_ca_types