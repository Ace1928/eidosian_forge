from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class AwsCredConfigGenerator(CredConfigGenerator):
    """The generator for AWS-based credential configs."""

    def __init__(self):
        super(AwsCredConfigGenerator, self).__init__(ConfigType.WORKLOAD_IDENTITY_POOLS)

    def get_token_type(self, subject_token_type):
        return 'urn:ietf:params:aws:token-type:aws4_request'

    def get_source(self, args):
        self._format_already_defined(args.credential_source_type)
        credential_source = {'environment_id': 'aws1', 'region_url': 'http://169.254.169.254/latest/meta-data/placement/availability-zone', 'url': 'http://169.254.169.254/latest/meta-data/iam/security-credentials', 'regional_cred_verification_url': 'https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15'}
        if args.enable_imdsv2:
            credential_source['imdsv2_session_token_url'] = 'http://169.254.169.254/latest/api/token'
        return credential_source