from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddContainerdConfigFlag(parser, hidden=True):
    """Adds --containerd-config-from-file flag to the given parser."""
    parser.add_argument('--containerd-config-from-file', type=arg_parsers.FileContents(), hidden=hidden, help='\nPath of the YAML/JSON file that contains the containerd configuration, including private registry access config.\n\nExample:\n    privateRegistryAccessConfig:\n      enabled: true\n      certificateAuthorityDomainConfig:\n        - gcpSecretManagerCertificateConfig:\n            secretURI: "projects/1234567890/secrets/my-cert/versions/2"\n          fqdns:\n            - "my.custom.domain"\n            - "10.2.3.4"\n\nFor detailed information on the configuration usage, please refer to DOC_LINK.\n\nNote: updating the containerd configuration of an existing cluster/node-pool requires recreation of the nodes which which might cause a disruption.\n')