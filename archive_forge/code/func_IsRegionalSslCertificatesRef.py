from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def IsRegionalSslCertificatesRef(ssl_certificate_ref):
    """Returns True if the SSL Certificate reference is regional."""
    return ssl_certificate_ref.Collection() == 'compute.regionSslCertificates'