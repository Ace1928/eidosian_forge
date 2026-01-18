from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddCommonEnterpriseCertConfigFlags(parser):
    """Common ECP configuration flags."""
    parser.add_argument('--ecp', default=None, help='Provide a custom path to the enterprise-certificate-proxy binary. This flag must be the full path to the binary.')
    parser.add_argument('--ecp-client', default=None, help='Provide a custom path to the enterprise-certificate-proxy shared client library. This flag must be the full path to the shared library.')
    parser.add_argument('--tls-offload', default=None, help='Provide a custom path to the enterprise-certificate-proxy shared tls offload library. This flag must be the full path to the shared library.')
    parser.add_argument('--output-file', default=None, help='Override the file path that the enterprise-certificate-proxy configuration is written to.')