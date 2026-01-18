import boto.exception
from boto.compat import json
import requests
import boto
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def _commit_without_auth(self, sdf, api_version):
    url = 'http://%s/%s/documents/batch' % (self.endpoint, api_version)
    session = requests.Session()
    session.proxies = self.proxy
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=5)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    resp = session.post(url, data=sdf, headers={'Content-Type': 'application/json'})
    return resp