from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def _GenerateSbomRefOccurrence(artifact, sbom, note, storage):
    """Create the SBOM reference note if not exists.

  Args:
    artifact: Artifact, the artifact metadata SBOM file generated from.
    sbom: SbomFile, metadata of the SBOM file.
    note: Note, the Note object we will use to attach occurrence.
    storage: str, the path that SBOM is stored remotely.

  Returns:
    An Occurrence object for the SBOM reference.
  """
    messages = ca_requests.GetMessages()
    sbom_digsets = messages.SbomReferenceIntotoPredicate.DigestValue()
    for k, v in sbom.digests.items():
        sbom_digsets.additionalProperties.append(messages.SbomReferenceIntotoPredicate.DigestValue.AdditionalProperty(key=k, value=v))
    predicate = messages.SbomReferenceIntotoPredicate(digest=sbom_digsets, location=storage, mimeType=sbom.GetMimeType(), referrerId=_SBOM_REFERENCE_REFERRERID)
    payload = messages.SbomReferenceIntotoPayload(predicateType=_SBOM_REFERENCE_PREDICATE_TYPE, _type=_SBOM_REFERENCE_TARGET_TYPE, predicate=predicate)
    artifact_digests = messages.Subject.DigestValue()
    for k, v in artifact.digests.items():
        artifact_digests.additionalProperties.append(messages.Subject.DigestValue.AdditionalProperty(key=k, value=v))
    sbom_subject = messages.Subject(digest=artifact_digests, name=artifact.resource_uri)
    payload.subject.append(sbom_subject)
    ref_occ = messages.SBOMReferenceOccurrence(payload=payload, payloadType=_SBOM_REFERENCE_PAYLOAD_TYPE)
    occ = messages.Occurrence(sbomReference=ref_occ, noteName=note.name, resourceUri=artifact.GetOccurrenceResourceUri())
    return occ