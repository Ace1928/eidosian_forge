from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _request_compute_metadata(path):
    req = urllib.request.Request('%s/computeMetadata/v1/%s' % (_gce_metadata_endpoint(), path), headers={'Metadata-Flavor': 'Google'})
    resp = urllib.request.urlopen(req)
    return _as_text(resp.read())