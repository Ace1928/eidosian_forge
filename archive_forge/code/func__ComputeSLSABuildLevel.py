from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _ComputeSLSABuildLevel(provenance):
    """Computes SLSA build level from a build provenance.

  Determines SLSA Level based on a list of occurrences,
  preferring data from SLSA v1.0 occurrences over others.

  Args:
    provenance: build provenance list containing build occurrences.

  Returns:
    A string `unknown` if build provenance doesn't exist, otherwise
    an integer from 0 to 3 indicating SLSA build level.
  """
    if not provenance:
        return 'unknown'
    builder_id_v1 = 'https://cloudbuild.googleapis.com/GoogleHostedWorker'
    builds_v1 = [p for p in provenance if p.build and p.build.inTotoSlsaProvenanceV1]
    for build_v1 in builds_v1:
        provenance_v1 = build_v1.build.inTotoSlsaProvenanceV1
        if provenance_v1.predicate and provenance_v1.predicate.runDetails and provenance_v1.predicate.runDetails.builder and provenance_v1.predicate.runDetails.builder.id and (provenance_v1.predicate.runDetails.builder.id == builder_id_v1):
            return 3
    builds_v0_1 = [p for p in provenance if p.build and p.build.intotoStatement]
    if not builds_v0_1:
        return 'unknown'
    provenance = builds_v0_1[0]
    intoto = provenance.build.intotoStatement
    if _HasSteps(intoto):
        if _HasValidKey(provenance):
            if _HasLevel3BuildVersion(intoto):
                return 3
            return 2
        return 1
    return 0