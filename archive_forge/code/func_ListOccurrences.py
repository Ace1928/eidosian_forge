from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.core import resources
def ListOccurrences(response, args):
    """Call CA APIs for vulnerabilities if --show-package-vulnerability is set."""
    if not args.show_package_vulnerability:
        return response
    project, maven_resource = _GenerateMavenResourceFromResponse(response)
    metadata = ca_util.GetMavenArtifactOccurrences(project, maven_resource)
    if metadata.ArtifactsDescribeView():
        response.update(metadata.ArtifactsDescribeView())
    else:
        response.update({'package_vulnerability_summary': 'No vulnerability data found.'})
    return response