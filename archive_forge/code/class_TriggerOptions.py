from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerOptions(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'sourceProvenanceHash': self.request.get('source_provenance_hash'), u'requestedVerifyOption': self.request.get('requested_verify_option'), u'machineType': self.request.get('machine_type'), u'diskSizeGb': self.request.get('disk_size_gb'), u'substitutionOption': self.request.get('substitution_option'), u'dynamicSubstitutions': self.request.get('dynamic_substitutions'), u'logStreamingOption': self.request.get('log_streaming_option'), u'workerPool': self.request.get('worker_pool'), u'logging': self.request.get('logging'), u'env': self.request.get('env'), u'secretEnv': self.request.get('secret_env'), u'volumes': TriggerVolumesArray(self.request.get('volumes', []), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'sourceProvenanceHash': self.request.get(u'sourceProvenanceHash'), u'requestedVerifyOption': self.request.get(u'requestedVerifyOption'), u'machineType': self.request.get(u'machineType'), u'diskSizeGb': self.request.get(u'diskSizeGb'), u'substitutionOption': self.request.get(u'substitutionOption'), u'dynamicSubstitutions': self.request.get(u'dynamicSubstitutions'), u'logStreamingOption': self.request.get(u'logStreamingOption'), u'workerPool': self.request.get(u'workerPool'), u'logging': self.request.get(u'logging'), u'env': self.request.get(u'env'), u'secretEnv': self.request.get(u'secretEnv'), u'volumes': TriggerVolumesArray(self.request.get(u'volumes', []), self.module).from_response()})