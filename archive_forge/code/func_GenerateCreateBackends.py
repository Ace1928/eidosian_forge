from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.metastore import validators as validator
import six
def GenerateCreateBackends(job_ref, args, create_federation_req):
    """Construct the backend names for create requests of Dataproc Metastore federations.

  Args:
    job_ref: A resource ref to the parsed Federation resource.
    args: The parsed args namespace from CLI.
    create_federation_req: Create federation request for the API call.

  Returns:
    Modified request for the API call.
  """
    return validator.ParseBackendsIntoRequest(job_ref, create_federation_req)