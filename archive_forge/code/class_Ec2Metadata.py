import json
import re
import socket
import time
import zlib
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
class Ec2Metadata:
    ec2_metadata_token_uri = 'http://169.254.169.254/latest/api/token'
    ec2_metadata_uri = 'http://169.254.169.254/latest/meta-data/'
    ec2_metadata_instance_tags_uri = 'http://169.254.169.254/latest/meta-data/tags/instance'
    ec2_sshdata_uri = 'http://169.254.169.254/latest/meta-data/public-keys/0/openssh-key'
    ec2_userdata_uri = 'http://169.254.169.254/latest/user-data/'
    ec2_dynamicdata_uri = 'http://169.254.169.254/latest/dynamic/'

    def __init__(self, module, ec2_metadata_token_uri=None, ec2_metadata_uri=None, ec2_metadata_instance_tags_uri=None, ec2_sshdata_uri=None, ec2_userdata_uri=None, ec2_dynamicdata_uri=None):
        self.module = module
        self.uri_token = ec2_metadata_token_uri or self.ec2_metadata_token_uri
        self.uri_meta = ec2_metadata_uri or self.ec2_metadata_uri
        self.uri_instance_tags = ec2_metadata_instance_tags_uri or self.ec2_metadata_instance_tags_uri
        self.uri_user = ec2_userdata_uri or self.ec2_userdata_uri
        self.uri_ssh = ec2_sshdata_uri or self.ec2_sshdata_uri
        self.uri_dynamic = ec2_dynamicdata_uri or self.ec2_dynamicdata_uri
        self._data = {}
        self._token = None
        self._prefix = 'ansible_ec2_%s'

    def _decode(self, data):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            self.module.warn('Decoding user-data as UTF-8 failed, return data as is ignoring any error')
            return data.decode('utf-8', errors='ignore')

    def decode_user_data(self, data):
        is_compressed = False
        if data.startswith(b'x\x9c') or data.startswith(b'\x1f\x8b'):
            is_compressed = True
        if is_compressed:
            try:
                decompressed = zlib.decompress(data, zlib.MAX_WBITS | 32)
                return self._decode(decompressed)
            except zlib.error:
                self.module.warn('Unable to decompress user-data using zlib, attempt to decode original user-data as UTF-8')
                return self._decode(data)
        else:
            return self._decode(data)

    def _fetch(self, url):
        encoded_url = quote(url, safe="%/:=&?~#+!$,;'@()*[]")
        headers = {}
        if self._token:
            headers = {'X-aws-ec2-metadata-token': self._token}
        response, info = fetch_url(self.module, encoded_url, headers=headers, force=True)
        if info.get('status') in (401, 403):
            self.module.fail_json(msg='Failed to retrieve metadata from AWS: {0}'.format(info['msg']), response=info)
        elif info.get('status') not in (200, 404):
            time.sleep(3)
            self.module.warn('Retrying query to metadata service. First attempt failed: {0}'.format(info['msg']))
            response, info = fetch_url(self.module, encoded_url, headers=headers, force=True)
            if info.get('status') not in (200, 404):
                self.module.fail_json(msg='Failed to retrieve metadata from AWS: {0}'.format(info['msg']), response=info)
        if response and info['status'] < 400:
            data = response.read()
            if 'user-data' in encoded_url:
                return to_text(self.decode_user_data(data))
        else:
            data = None
        return to_text(data)

    def _mangle_fields(self, fields, uri, filter_patterns=None):
        filter_patterns = ['public-keys-0'] if filter_patterns is None else filter_patterns
        new_fields = {}
        for key, value in fields.items():
            split_fields = key[len(uri):].split('/')
            if len(split_fields) == 3 and split_fields[0:2] == ['iam', 'security-credentials'] and (':' not in split_fields[2]):
                new_fields[self._prefix % 'iam-instance-profile-role'] = split_fields[2]
            if len(split_fields) > 1 and split_fields[1]:
                new_key = '-'.join(split_fields)
                new_fields[self._prefix % new_key] = value
            else:
                new_key = ''.join(split_fields)
                new_fields[self._prefix % new_key] = value
        for pattern in filter_patterns:
            for key in dict(new_fields):
                match = re.search(pattern, key)
                if match:
                    new_fields.pop(key)
        return new_fields

    def fetch(self, uri, recurse=True):
        raw_subfields = self._fetch(uri)
        if not raw_subfields:
            return
        subfields = raw_subfields.split('\n')
        for field in subfields:
            if field.endswith('/') and recurse:
                self.fetch(uri + field)
            if uri.endswith('/'):
                new_uri = uri + field
            else:
                new_uri = uri + '/' + field
            if new_uri not in self._data and (not new_uri.endswith('/')):
                content = self._fetch(new_uri)
                if field == 'security-groups' or field == 'security-group-ids':
                    sg_fields = ','.join(content.split('\n'))
                    self._data['%s' % new_uri] = sg_fields
                else:
                    try:
                        json_dict = json.loads(content)
                        self._data['%s' % new_uri] = content
                        for key, value in json_dict.items():
                            self._data['%s:%s' % (new_uri, key.lower())] = value
                    except (json_decode_error, AttributeError):
                        self._data['%s' % new_uri] = content

    def fix_invalid_varnames(self, data):
        """Change ':'' and '-' to '_' to ensure valid template variable names"""
        new_data = data.copy()
        for key, value in data.items():
            if ':' in key or '-' in key:
                newkey = re.sub(':|-', '_', key)
                new_data[newkey] = value
                del new_data[key]
        return new_data

    def fetch_session_token(self, uri_token):
        """Used to get a session token for IMDSv2"""
        headers = {'X-aws-ec2-metadata-token-ttl-seconds': '60'}
        response, info = fetch_url(self.module, uri_token, method='PUT', headers=headers, force=True)
        if info.get('status') == 403:
            self.module.fail_json(msg='Failed to retrieve metadata token from AWS: {0}'.format(info['msg']), response=info)
        elif info.get('status') not in (200, 404):
            time.sleep(3)
            self.module.warn('Retrying query to metadata service. First attempt failed: {0}'.format(info['msg']))
            response, info = fetch_url(self.module, uri_token, method='PUT', headers=headers, force=True)
            if info.get('status') not in (200, 404):
                self.module.fail_json(msg='Failed to retrieve metadata token from AWS: {0}'.format(info['msg']), response=info)
        if response:
            token_data = response.read()
        else:
            token_data = None
        return to_text(token_data)

    def run(self):
        self._token = self.fetch_session_token(self.uri_token)
        self.fetch(self.uri_meta)
        data = self._mangle_fields(self._data, self.uri_meta)
        data[self._prefix % 'user-data'] = self._fetch(self.uri_user)
        data[self._prefix % 'public-key'] = self._fetch(self.uri_ssh)
        self._data = {}
        self.fetch(self.uri_dynamic)
        dyndata = self._mangle_fields(self._data, self.uri_dynamic)
        data.update(dyndata)
        data = self.fix_invalid_varnames(data)
        instance_tags_keys = self._fetch(self.uri_instance_tags)
        instance_tags_keys = instance_tags_keys.split('\n') if instance_tags_keys != 'None' else []
        data[self._prefix % 'instance_tags_keys'] = instance_tags_keys
        if 'ansible_ec2_instance_identity_document_region' in data:
            data['ansible_ec2_placement_region'] = data['ansible_ec2_instance_identity_document_region']
        return data