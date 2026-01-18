from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddProxySecretName(parser, required=False):
    help_txt = '\nName of the Kubernetes secret that contains the HTTP/HTTPS\nproxy configuration.\n'
    parser.add_argument('--proxy-secret-name', required=required, help=help_txt)