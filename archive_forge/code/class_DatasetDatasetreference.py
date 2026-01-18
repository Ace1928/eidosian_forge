from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class DatasetDatasetreference(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'datasetId': self.request.get('dataset_id'), u'projectId': self.request.get('project_id')})

    def from_response(self):
        return remove_nones_from_dict({u'datasetId': self.module.params.get('dataset_id'), u'projectId': self.module.params.get('project_id')})