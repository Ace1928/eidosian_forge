from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
def CreateFromName(name):
    """Creates a client and resource reference for a name.

  Args:
    name: An operation name, optionally including projects/, operations/, and a
        project name.

  Returns:
    A tuple containing the genomics client and resource reference.
  """
    client = None
    if re.search('[a-zA-Z]', name.split('/')[-1]):
        client = GenomicsV1ApiClient()
    else:
        client = GenomicsV2ApiClient()
    return (client, client.ResourceFromName(name))