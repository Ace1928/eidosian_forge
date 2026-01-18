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
def _MakeNote(vuln, status, product, publisher, document, msgs):
    """Makes a note.

  Args:
    vuln: vulnerability proto
    status: string of status of vulnerability
    product: product proto
    publisher: publisher proto.
    document: document proto.
    msgs: container analysis messages

  Returns:
    noteid, and note
  """
    state = None
    remediations = []
    desc_note = None
    justification = None
    notes = vuln.get('notes')
    if notes is not None:
        for note in notes:
            if note['category'] == 'description':
                desc_note = note
    if status == 'known_affected':
        state = msgs.Assessment.StateValueValuesEnum.AFFECTED
        remediations = _GetRemediations(vuln, product, msgs)
    elif status == 'known_not_affected':
        state = msgs.Assessment.StateValueValuesEnum.NOT_AFFECTED
        justification = _GetJustifications(vuln, product, msgs)
    elif status == 'fixed':
        state = msgs.Assessment.StateValueValuesEnum.FIXED
    elif status == 'under_investigation':
        state = msgs.Assessment.StateValueValuesEnum.UNDER_INVESTIGATION
    note = msgs.Note(vulnerabilityAssessment=msgs.VulnerabilityAssessmentNote(title=document['title'], publisher=publisher, product=product, assessment=msgs.Assessment(vulnerabilityId=vuln['cve'], shortDescription=desc_note['title'] if desc_note is not None else None, longDescription=desc_note['text'] if desc_note is not None else None, state=state, remediations=remediations, justification=justification)))
    key = note.vulnerabilityAssessment.product.genericUri + note.vulnerabilityAssessment.assessment.vulnerabilityId
    result = hashlib.md5(key.encode())
    noteid = result.hexdigest()
    return (noteid, note)