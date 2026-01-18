from __future__ import (absolute_import, division, print_function)
import logging
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
class UMRestAPI(object):
    """ send REST request and process response """

    def __init__(self, module, timeout=60):
        self.module = module
        self.username = self.module.params['username']
        self.password = self.module.params['password']
        self.hostname = self.module.params['hostname']
        self.verify = self.module.params['validate_certs']
        self.max_records = self.module.params['max_records']
        self.timeout = timeout
        if self.module.params.get('http_port') is not None:
            self.url = 'https://%s:%d' % (self.hostname, self.module.params['http_port'])
        else:
            self.url = 'https://%s' % self.hostname
        self.errors = list()
        self.debug_logs = list()
        self.check_required_library()
        if has_feature(module, 'trace_apis'):
            logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')

    def check_required_library(self):
        if not HAS_REQUESTS:
            self.module.fail_json(msg=missing_required_lib('requests'))

    def get_records(self, message, api):
        records = list()
        try:
            if message['total_records'] > 0:
                records = message['records']
                if message['total_records'] != len(records):
                    self.module.warn('Mismatch between received: %d and expected: %d records.' % (len(records), message['total_records']))
        except KeyError as exc:
            self.module.fail_json(msg='Error: unexpected response from %s: %s - expecting key: %s' % (api, message, to_native(exc)))
        return records

    def send_request(self, method, api, params, json=None, accept=None):
        """ send http request and process response, including error conditions """
        url = self.url + api
        status_code = None
        content = None
        json_dict = None
        json_error = None
        error_details = None
        headers = None
        if accept is not None:
            headers = dict()
            if accept is not None:
                headers['accept'] = accept

        def check_contents(response):
            """json() may fail on an empty value, but it's OK if no response is expected.
               To avoid false positives, only report an issue when we expect to read a value.
               The first get will see it.
            """
            if method == 'GET' and has_feature(self.module, 'strict_json_check'):
                contents = response.content
                if len(contents) > 0:
                    raise ValueError('Expecting json, got: %s' % contents)

        def get_json(response):
            """ extract json, and error message if present """
            try:
                json = response.json()
            except ValueError:
                check_contents(response)
                return (None, None)
            error = json.get('error')
            return (json, error)
        self.log_debug('sending', repr(dict(method=method, url=url, verify=self.verify, params=params, timeout=self.timeout, json=json, headers=headers)))
        try:
            response = requests.request(method, url, verify=self.verify, auth=(self.username, self.password), params=params, timeout=self.timeout, json=json, headers=headers)
            content = response.content
            status_code = response.status_code
            response.raise_for_status()
            json_dict, json_error = get_json(response)
        except requests.exceptions.HTTPError as err:
            __, json_error = get_json(response)
            if json_error is None:
                self.log_error(status_code, 'HTTP error: %s' % err)
                error_details = str(err)
        except requests.exceptions.ConnectionError as err:
            self.log_error(status_code, 'Connection error: %s' % err)
            error_details = str(err)
        except Exception as err:
            self.log_error(status_code, 'Other error: %s' % err)
            error_details = str(err)
        if json_error is not None:
            self.log_error(status_code, 'Endpoint error: %d: %s' % (status_code, json_error))
            error_details = json_error
        self.log_debug(status_code, content)
        return (json_dict, error_details)

    def get(self, api, params):

        def get_next_api(message):
            """make sure _links is present, and href is present if next is present
               return api if next is present, None otherwise
               return error if _links or href are missing
            """
            api, error = (None, None)
            if message is None or '_links' not in message:
                error = 'Expecting _links key in %s' % message
            elif 'next' in message['_links']:
                if 'href' in message['_links']['next']:
                    api = message['_links']['next']['href']
                else:
                    error = 'Expecting href key in %s' % message['_links']['next']
            return (api, error)
        method = 'GET'
        records = list()
        if self.max_records is not None:
            if params and 'max_records' not in params:
                params['max_records'] = self.max_records
            else:
                params = dict(max_records=self.max_records)
        api = '/api/%s' % api
        while api:
            message, error = self.send_request(method, api, params)
            if error:
                return (message, error)
            api, error = get_next_api(message)
            if error:
                return (message, error)
            if 'records' in message:
                records.extend(message['records'])
            params = None
        if records:
            message['records'] = records
        return (message, error)

    def log_error(self, status_code, message):
        LOG.error('%s: %s', status_code, message)
        self.errors.append(message)
        self.debug_logs.append((status_code, message))

    def log_debug(self, status_code, content):
        LOG.debug('%s: %s', status_code, content)
        self.debug_logs.append((status_code, content))