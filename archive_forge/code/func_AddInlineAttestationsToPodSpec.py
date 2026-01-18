from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def AddInlineAttestationsToPodSpec(pod_spec, attestations):
    """Inlines attestations into a Kubernetes PodSpec.

  Args:
    pod_spec: The PodSpec provided by the user.
    attestations: List of attestations returned by the policy evaluator in comma
      separated DSSE form.

  Returns:
    Modified PodSpec with attestations inlined.
  """
    annotations = pod_spec['metadata'].get('annotations', {})
    existing_attestations = annotations.get(_BINAUTHZ_ATTESTATION_ANNOTATION_KEY, None)
    if existing_attestations:
        annotations[_BINAUTHZ_ATTESTATION_ANNOTATION_KEY] = ','.join([existing_attestations] + attestations)
    else:
        annotations[_BINAUTHZ_ATTESTATION_ANNOTATION_KEY] = ','.join(attestations)
    pod_spec['metadata']['annotations'] = annotations
    return pod_spec