import boto.exception
from boto.compat import json
import requests
import boto
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def _commit_with_auth(self, sdf, api_version):
    return self.domain_connection.upload_documents(sdf, 'application/json')