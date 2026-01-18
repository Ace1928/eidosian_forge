from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.core import log
from  googlecloudsdk.core.util.files import FileReader
def _GetRemediations(vuln, product, msgs):
    """Get remediations.

  Args:
    vuln: vulnerability proto
    product: product proto
    msgs: container analysis messages

  Returns:
    remediations proto
  """
    remediations = []
    vuln_remediations = vuln.get('remediations')
    if vuln_remediations is None:
        return remediations
    for remediation in vuln_remediations:
        remediation_type = remediation['category']
        remediation_detail = remediation['details']
        remediation_enum = msgs.Remediation.RemediationTypeValueValuesEnum.lookup_by_name(remediation_type.upper())
        for product_id in remediation['product_ids']:
            if product_id == product.id:
                remediation = msgs.Remediation(remediationType=remediation_enum, details=remediation_detail)
                remediations.append(remediation)
    return remediations