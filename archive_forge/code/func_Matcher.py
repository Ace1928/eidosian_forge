from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def Matcher(gcloud_insight):
    matches_member = False
    matches_role = False
    for additional_property in gcloud_insight.content.additionalProperties:
        if additional_property.key == 'risk':
            for p in additional_property.value.object_value.properties:
                if p.key == 'usageAtRisk':
                    for f in p.value.object_value.properties:
                        if f.key == 'iamPolicyUtilization':
                            for iam_p in f.value.object_value.properties:
                                if iam_p.key == 'member':
                                    if iam_p.value.string_value == member:
                                        matches_member = True
                                if iam_p.key == 'role':
                                    if iam_p.value.string_value == role:
                                        matches_role = True
    return matches_member and matches_role