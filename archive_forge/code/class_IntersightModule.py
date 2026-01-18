from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
class IntersightModule:

    def __init__(self, module):
        self.module = module
        self.result = dict(changed=False)
        if not HAS_CRYPTOGRAPHY:
            self.module.fail_json(msg='cryptography is required for this module')
        self.host = self.module.params['api_uri']
        self.public_key = self.module.params['api_key_id']
        try:
            with open(self.module.params['api_private_key'], 'r') as f:
                self.private_key = f.read()
        except (FileNotFoundError, OSError):
            self.private_key = self.module.params['api_private_key']
        self.digest_algorithm = ''
        self.response_list = []

    def get_sig_b64encode(self, data):
        """
        Generates a signed digest from a String

        :param digest: string to be signed & hashed
        :return: instance of digest object
        """
        r = re.compile('\\s*-----BEGIN (.*)-----\\s+')
        m = r.match(self.private_key)
        if not m:
            raise ValueError('Not a valid PEM pre boundary')
        pem_header = m.group(1)
        key = serialization.load_pem_private_key(self.private_key.encode(), None, default_backend())
        if pem_header == 'RSA PRIVATE KEY':
            sign = key.sign(data.encode(), padding.PKCS1v15(), hashes.SHA256())
            self.digest_algorithm = 'rsa-sha256'
        elif pem_header == 'EC PRIVATE KEY':
            sign = key.sign(data.encode(), ec.ECDSA(hashes.SHA256()))
            self.digest_algorithm = 'hs2019'
        else:
            raise Exception('Unsupported key: {0}'.format(pem_header))
        return b64encode(sign)

    def get_auth_header(self, hdrs, signed_msg):
        """
        Assmebled an Intersight formatted authorization header

        :param hdrs : object with header keys
        :param signed_msg: base64 encoded sha256 hashed body
        :return: concatenated authorization header
        """
        auth_str = 'Signature'
        auth_str = auth_str + ' ' + 'keyId="' + self.public_key + '",' + 'algorithm="' + self.digest_algorithm + '",'
        auth_str = auth_str + 'headers="(request-target)'
        for key, dummy in hdrs.items():
            auth_str = auth_str + ' ' + key.lower()
        auth_str = auth_str + '"'
        auth_str = auth_str + ',' + 'signature="' + signed_msg.decode('ascii') + '"'
        return auth_str

    def get_moid_by_name(self, resource_path, target_name):
        """
        Retrieve an Intersight object moid by name

        :param resource_path: intersight resource path e.g. '/ntp/Policies'
        :param target_name: intersight object name
        :return: json http response object
        """
        query_params = {'$filter': "Name eq '{0}'".format(target_name)}
        options = {'http_method': 'GET', 'resource_path': resource_path, 'query_params': query_params}
        get_moid = self.intersight_call(**options)
        if get_moid.json()['Results'] is not None:
            located_moid = get_moid.json()['Results'][0]['Moid']
        else:
            raise KeyError('Intersight object with name "{0}" not found!'.format(target_name))
        return located_moid

    def call_api(self, **options):
        """
        Call the Intersight API and check for success status
        :param options: options dict with method and other params for API call
        :return: json http response object
        """
        try:
            response, info = self.intersight_call(**options)
            if not re.match('2..', str(info['status'])):
                raise RuntimeError(info['status'], info['msg'], info['body'])
        except Exception as e:
            self.module.fail_json(msg='API error: %s ' % str(e))
        response_data = response.read()
        if len(response_data) > 0:
            resp_json = json.loads(response_data)
            resp_json['trace_id'] = info.get('x-starship-traceid')
            return resp_json
        return {}

    def intersight_call(self, http_method='', resource_path='', query_params=None, body=None, moid=None, name=None):
        """
        Invoke the Intersight API

        :param resource_path: intersight resource path e.g. '/ntp/Policies'
        :param query_params: dictionary object with query string parameters as key/value pairs
        :param body: dictionary object with intersight data
        :param moid: intersight object moid
        :param name: intersight object name
        :return: json http response object
        """
        target_host = urlparse(self.host).netloc
        target_path = urlparse(self.host).path
        query_path = ''
        method = http_method.upper()
        bodyString = ''
        if method not in ['GET', 'POST', 'PATCH', 'DELETE']:
            raise ValueError('Please select a valid HTTP verb (GET/POST/PATCH/DELETE)')
        if resource_path != '' and (not (resource_path, str)):
            raise TypeError('The *resource_path* value is required and must be of type "<str>"')
        if query_params is not None and (not isinstance(query_params, dict)):
            raise TypeError('The *query_params* value must be of type "<dict>"')
        if moid is not None and len(moid.encode('utf-8')) != 24:
            raise ValueError('Invalid *moid* value!')
        if query_params:
            query_path = '?' + urlencode(query_params)
        if method in ('PATCH', 'DELETE'):
            if moid is None:
                if name is not None:
                    if isinstance(name, str):
                        moid = self.get_moid_by_name(resource_path, name)
                    else:
                        raise TypeError('The *name* value must be of type "<str>"')
                else:
                    raise ValueError('Must set either *moid* or *name* with "PATCH/DELETE!"')
        if moid is not None:
            resource_path += '/' + moid
        if method != 'GET':
            bodyString = json.dumps(body)
        target_url = self.host + resource_path + query_path
        request_target = method.lower() + ' ' + target_path + resource_path + query_path
        cdate = get_gmt_date()
        body_digest = get_sha256_digest(bodyString)
        b64_body_digest = b64encode(body_digest.digest())
        auth_header = {'Host': target_host, 'Date': cdate, 'Digest': 'SHA-256=' + b64_body_digest.decode('ascii')}
        string_to_sign = prepare_str_to_sign(request_target, auth_header)
        b64_signed_msg = self.get_sig_b64encode(string_to_sign)
        auth_header = self.get_auth_header(auth_header, b64_signed_msg)
        request_header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Host': '{0}'.format(target_host), 'Date': '{0}'.format(cdate), 'Digest': 'SHA-256={0}'.format(b64_body_digest.decode('ascii')), 'Authorization': '{0}'.format(auth_header)}
        response, info = fetch_url(self.module, target_url, data=bodyString, headers=request_header, method=method, use_proxy=self.module.params['use_proxy'])
        return (response, info)

    def get_resource(self, resource_path, query_params, return_list=False):
        """
        GET a resource and return the 1st element found or the full Results list
        """
        options = {'http_method': 'get', 'resource_path': resource_path, 'query_params': query_params}
        response = self.call_api(**options)
        if response.get('Results'):
            if return_list:
                self.result['api_response'] = response['Results']
            else:
                self.result['api_response'] = response['Results'][0]
        self.result['trace_id'] = response.get('trace_id')

    def configure_resource(self, moid, resource_path, body, query_params, update_method=''):
        if not self.module.check_mode:
            if moid and update_method != 'post':
                options = {'http_method': 'patch', 'resource_path': resource_path, 'body': body, 'moid': moid}
                response_dict = self.call_api(**options)
                if response_dict.get('Results'):
                    self.result['api_response'] = response_dict['Results'][0]
                    self.result['trace_id'] = response_dict.get('trace_id')
            else:
                options = {'http_method': 'post', 'resource_path': resource_path, 'body': body}
                response_dict = self.call_api(**options)
                if response_dict:
                    self.result['api_response'] = response_dict
                    self.result['trace_id'] = response_dict.get('trace_id')
                elif query_params:
                    self.get_resource(resource_path=resource_path, query_params=query_params)
        self.result['changed'] = True

    def delete_resource(self, moid, resource_path):
        if not self.module.check_mode:
            options = {'http_method': 'delete', 'resource_path': resource_path, 'moid': moid}
            resp = self.call_api(**options)
            self.result['api_response'] = {}
            self.result['trace_id'] = resp.get('trace_id')
        self.result['changed'] = True

    def configure_policy_or_profile(self, resource_path):
        organization_moid = None
        self.get_resource(resource_path='/organization/Organizations', query_params={'$filter': "Name eq '" + self.module.params['organization'] + "'", '$select': 'Moid'})
        if self.result['api_response'].get('Moid'):
            organization_moid = self.result['api_response']['Moid']
        self.result['api_response'] = {}
        filter_str = "Name eq '" + self.module.params['name'] + "'"
        filter_str += "and Organization.Moid eq '" + organization_moid + "'"
        self.get_resource(resource_path=resource_path, query_params={'$filter': filter_str, '$expand': 'Organization'})
        moid = None
        resource_values_match = False
        if self.result['api_response'].get('Moid'):
            moid = self.result['api_response']['Moid']
            if self.module.params['state'] == 'present':
                resource_values_match = compare_values(self.api_body, self.result['api_response'])
            else:
                self.delete_resource(moid=moid, resource_path=resource_path)
                moid = None
        if self.module.params['state'] == 'present' and (not resource_values_match):
            self.api_body.pop('Organization')
            if not moid:
                self.api_body['Organization'] = {'Moid': organization_moid}
            self.configure_resource(moid=moid, resource_path=resource_path, body=self.api_body, query_params={'$filter': filter_str})
            if self.result['api_response'].get('Moid'):
                moid = self.result['api_response']['Moid']
        return moid