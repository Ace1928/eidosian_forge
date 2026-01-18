from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def RecommenderClient(api_version):
    return apis.GetClientInstance(RECOMMENDER_API_NAME, api_version)