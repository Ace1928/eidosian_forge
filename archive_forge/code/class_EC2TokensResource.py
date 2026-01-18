import urllib.parse
import flask
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_serialization import jsonutils
from keystone.api._shared import EC2_S3_Resource
from keystone.api._shared import json_home_relations
from keystone.common import render_token
from keystone.common import utils
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class EC2TokensResource(EC2_S3_Resource.ResourceBase):

    @staticmethod
    def _check_signature(creds_ref, credentials):
        signer = ec2_utils.Ec2Signer(creds_ref['secret'])
        signature = signer.generate(credentials)
        if credentials.get('signature'):
            if utils.auth_str_equal(credentials['signature'], signature):
                return True
            elif ':' in credentials['host']:
                parsed = urllib.parse.urlsplit('//' + credentials['host'])
                credentials['host'] = parsed.hostname
                signer = ec2_utils.Ec2Signer(creds_ref['secret'])
                signature = signer.generate(credentials)
                if utils.auth_str_equal(credentials['signature'], signature):
                    return True
            raise exception.Unauthorized(_('Invalid EC2 signature.'))
        else:
            raise exception.Unauthorized(_('EC2 signature not supplied.'))

    @ks_flask.unenforced_api
    def post(self):
        """Authenticate ec2 token.

        POST /v3/ec2tokens
        """
        token = self.handle_authenticate()
        token_reference = render_token.render_token_response_from_model(token)
        resp_body = jsonutils.dumps(token_reference)
        response = flask.make_response(resp_body, http.client.OK)
        response.headers['X-Subject-Token'] = token.id
        response.headers['Content-Type'] = 'application/json'
        return response