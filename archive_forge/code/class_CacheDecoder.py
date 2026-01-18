import json
import os
from collections import defaultdict
import hashlib
import tempfile
from functools import partial
import kubernetes.dynamic
import kubernetes.dynamic.discovery
from kubernetes import __version__
from kubernetes.dynamic.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.client.resource import (
class CacheDecoder(json.JSONDecoder):

    def __init__(self, client, *args, **kwargs):
        self.client = client
        json.JSONDecoder.__init__(self, *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj):
        if '_type' not in obj:
            return obj
        _type = obj.pop('_type')
        if _type == 'Resource':
            return kubernetes.dynamic.Resource(client=self.client, **obj)
        elif _type == 'ResourceList':
            return ResourceList(self.client, **obj)
        elif _type == 'ResourceGroup':
            return kubernetes.dynamic.discovery.ResourceGroup(obj['preferred'], resources=self.object_hook(obj['resources']))
        return obj