from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def IsGlobalSslCertificatesRef(ssl_certificate_ref):
    """Returns True if the SSL Certificate reference is global."""
    return ssl_certificate_ref.Collection() == 'compute.sslCertificates'