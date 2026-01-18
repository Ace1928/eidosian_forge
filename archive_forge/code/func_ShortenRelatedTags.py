from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.core import resources
def ShortenRelatedTags(response, unused_args):
    """Convert the tag resources into tag IDs."""
    tags = []
    for t in response.relatedTags:
        tag = resources.REGISTRY.ParseRelativeName(t.name, 'artifactregistry.projects.locations.repositories.packages.tags')
        tags.append(tag.tagsId)
    json_obj = json.loads(protojson.encode_message(response))
    json_obj.pop('relatedTags', None)
    if tags:
        json_obj['relatedTags'] = tags
    if response.metadata is not None:
        json_obj['metadata'] = {prop.key: prop.value.string_value for prop in response.metadata.additionalProperties}
    return json_obj