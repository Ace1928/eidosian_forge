from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class IscsiTarget(object):

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=False, aliases=['alias']), ping=dict(type='bool', required=False, default=True), chap_secret=dict(type='str', required=False, aliases=['chap', 'password'], no_log=True), unnamed_discovery=dict(type='bool', required=False, default=True), log_path=dict(type='str', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        args = self.module.params
        self.name = args['name']
        self.ping = args['ping']
        self.chap_secret = args['chap_secret']
        self.unnamed_discovery = args['unnamed_discovery']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'])
        self.check_mode = self.module.check_mode
        self.post_body = dict()
        self.controllers = list()
        log_path = args['log_path']
        self._logger = logging.getLogger(self.__class__.__name__)
        if log_path:
            logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(relativeCreated)dms %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n %(message)s')
        if not self.url.endswith('/'):
            self.url += '/'
        if self.chap_secret:
            if len(self.chap_secret) < 12 or len(self.chap_secret) > 57:
                self.module.fail_json(msg='The provided CHAP secret is not valid, it must be between 12 and 57 characters in length.')
            for c in self.chap_secret:
                ordinal = ord(c)
                if ordinal < 32 or ordinal > 126:
                    self.module.fail_json(msg='The provided CHAP secret is not valid, it may only utilize ascii characters with decimal values between 32 and 126.')

    @property
    def target(self):
        """Provide information on the iSCSI Target configuration

        Sample:
        {
          'alias': 'myCustomName',
          'ping': True,
          'unnamed_discovery': True,
          'chap': False,
          'iqn': 'iqn.1992-08.com.netapp:2800.000a132000b006d2000000005a0e8f45',
        }
        """
        target = dict()
        try:
            rc, data = request(self.url + 'storage-systems/%s/graph/xpath-filter?query=/storagePoolBundle/target' % self.ssid, headers=HEADERS, **self.creds)
            if not data:
                self.module.fail_json(msg="This storage-system doesn't appear to have iSCSI interfaces. Array Id [%s]." % self.ssid)
            data = data[0]
            chap = any([auth for auth in data['configuredAuthMethods']['authMethodData'] if auth['authMethod'] == 'chap'])
            target.update(dict(alias=data['alias']['iscsiAlias'], iqn=data['nodeName']['iscsiNodeName'], chap=chap))
            rc, data = request(self.url + 'storage-systems/%s/graph/xpath-filter?query=/sa/iscsiEntityData' % self.ssid, headers=HEADERS, **self.creds)
            data = data[0]
            target.update(dict(ping=data['icmpPingResponseEnabled'], unnamed_discovery=data['unnamedDiscoverySessionsEnabled']))
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve the iSCSI target information. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return target

    def apply_iscsi_settings(self):
        """Update the iSCSI target alias and CHAP settings"""
        update = False
        target = self.target
        body = dict()
        if self.name is not None and self.name != target['alias']:
            update = True
            body['alias'] = self.name
        if self.chap_secret:
            update = True
            body.update(dict(enableChapAuthentication=True, chapSecret=self.chap_secret))
        elif target['chap']:
            update = True
            body.update(dict(enableChapAuthentication=False))
        if update and (not self.check_mode):
            try:
                request(self.url + 'storage-systems/%s/iscsi/target-settings' % self.ssid, method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
            except Exception as err:
                self.module.fail_json(msg='Failed to update the iSCSI target settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def apply_target_changes(self):
        update = False
        target = self.target
        body = dict()
        if self.ping != target['ping']:
            update = True
            body['icmpPingResponseEnabled'] = self.ping
        if self.unnamed_discovery != target['unnamed_discovery']:
            update = True
            body['unnamedDiscoverySessionsEnabled'] = self.unnamed_discovery
        self._logger.info(pformat(body))
        if update and (not self.check_mode):
            try:
                request(self.url + 'storage-systems/%s/iscsi/entity' % self.ssid, method='POST', data=json.dumps(body), timeout=60, headers=HEADERS, **self.creds)
            except Exception as err:
                self.module.fail_json(msg='Failed to update the iSCSI target settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def update(self):
        update = self.apply_iscsi_settings()
        update = self.apply_target_changes() or update
        target = self.target
        data = dict(((key, target[key]) for key in target if key in ['iqn', 'alias']))
        self.module.exit_json(msg='The interface settings have been updated.', changed=update, **data)

    def __call__(self, *args, **kwargs):
        self.update()