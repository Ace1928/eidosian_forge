from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceCdnpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'cacheKeyPolicy': RegionBackendServiceCachekeypolicy(self.request.get('cache_key_policy', {}), self.module).to_request(), u'signedUrlCacheMaxAgeSec': self.request.get('signed_url_cache_max_age_sec'), u'defaultTtl': self.request.get('default_ttl'), u'maxTtl': self.request.get('max_ttl'), u'clientTtl': self.request.get('client_ttl'), u'negativeCaching': self.request.get('negative_caching'), u'negativeCachingPolicy': RegionBackendServiceNegativecachingpolicyArray(self.request.get('negative_caching_policy', []), self.module).to_request(), u'cacheMode': self.request.get('cache_mode'), u'serveWhileStale': self.request.get('serve_while_stale')})

    def from_response(self):
        return remove_nones_from_dict({u'cacheKeyPolicy': RegionBackendServiceCachekeypolicy(self.request.get(u'cacheKeyPolicy', {}), self.module).from_response(), u'signedUrlCacheMaxAgeSec': self.request.get(u'signedUrlCacheMaxAgeSec'), u'defaultTtl': self.request.get(u'defaultTtl'), u'maxTtl': self.request.get(u'maxTtl'), u'clientTtl': self.request.get(u'clientTtl'), u'negativeCaching': self.request.get(u'negativeCaching'), u'negativeCachingPolicy': RegionBackendServiceNegativecachingpolicyArray(self.request.get(u'negativeCachingPolicy', []), self.module).from_response(), u'cacheMode': self.request.get(u'cacheMode'), u'serveWhileStale': self.request.get(u'serveWhileStale')})