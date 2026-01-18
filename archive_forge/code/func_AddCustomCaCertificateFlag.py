from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def AddCustomCaCertificateFlag(parser):
    """Add the Custom CA Certificates flag.

  Args:
    parser: An argparse parser that you can use to add arguments that go on the
      command line after this command. Positional arguments are allowed.
  """
    parser.add_argument('--custom-certificate-authority-roots', help="                      Comma-separated list of CA root certificates to use when\n                      connecting to the type's API by HTTPS.", type=arg_parsers.ArgList(min_length=1), default=[], metavar='CA_ROOT')