import os
from boto.compat import six
from datetime import datetime
import boto
from boto import config
from boto.compat import expanduser
from boto.pyami.config import Config
from boto.exception import InvalidInstanceMetadataError
from boto.gs.acl import ACL
from boto.gs.acl import CannedACLStrings as CannedGSACLStrings
from boto.s3.acl import CannedACLStrings as CannedS3ACLStrings
from boto.s3.acl import Policy
def _get_credentials_from_metadata(self, metadata):
    creds = list(metadata.values())[0]
    if not isinstance(creds, dict):
        if creds == '':
            msg = 'an empty string'
        else:
            msg = 'type: %s' % creds
        raise InvalidInstanceMetadataError('Expected a dict type of credentials instead received %s' % msg)
    try:
        access_key = creds['AccessKeyId']
        secret_key = self._convert_key_to_str(creds['SecretAccessKey'])
        security_token = creds['Token']
        expires_at = creds['Expiration']
    except KeyError as e:
        raise InvalidInstanceMetadataError('Credentials from instance metadata missing required key: %s' % e)
    return (access_key, secret_key, security_token, expires_at)