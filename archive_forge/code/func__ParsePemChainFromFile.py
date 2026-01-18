from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import create_utils
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import pem_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _ParsePemChainFromFile(self, pem_chain_file):
    """Parses a pem chain from a file, splitting the leaf cert and chain.

    Args:
      pem_chain_file: file containing the pem_chain.

    Raises:
      exceptions.InvalidArgumentException if not enough certificates are
      included.

    Returns:
      A tuple with (leaf_cert, rest_of_chain)
    """
    try:
        pem_chain_input = files.ReadFileContents(pem_chain_file)
    except (files.Error, OSError, IOError):
        raise exceptions.BadFileException("Could not read provided PEM chain file '{}'.".format(pem_chain_file))
    certs = pem_utils.ValidateAndParsePemChain(pem_chain_input)
    if len(certs) < 2:
        raise exceptions.InvalidArgumentException('pem-chain', 'The pem_chain must include at least two certificates - the subordinate CA certificate and an issuer certificate.')
    return (certs[0], certs[1:])