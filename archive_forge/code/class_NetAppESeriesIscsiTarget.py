from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
class NetAppESeriesIscsiTarget(NetAppESeriesModule):

    def __init__(self):
        ansible_options = dict(name=dict(type='str', required=False, aliases=['alias']), ping=dict(type='bool', required=False, default=True), chap_secret=dict(type='str', required=False, aliases=['chap', 'password'], no_log=True), unnamed_discovery=dict(type='bool', required=False, default=True))
        super(NetAppESeriesIscsiTarget, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', supports_check_mode=True)
        args = self.module.params
        self.name = args['name']
        self.ping = args['ping']
        self.chap_secret = args['chap_secret']
        self.unnamed_discovery = args['unnamed_discovery']
        self.check_mode = self.module.check_mode
        self.post_body = dict()
        self.controllers = list()
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
          "alias": "myCustomName",
          "ping": True,
          "unnamed_discovery": True,
          "chap": False,
          "iqn": "iqn.1992-08.com.netapp:2800.000a132000b006d2000000005a0e8f45",
        }
        """
        target = dict()
        try:
            rc, data = self.request('storage-systems/%s/graph/xpath-filter?query=/storagePoolBundle/target' % self.ssid)
            if not data:
                self.module.fail_json(msg='This storage-system does not appear to have iSCSI interfaces. Array Id [%s].' % self.ssid)
            data = data[0]
            chap = any([auth for auth in data['configuredAuthMethods']['authMethodData'] if auth['authMethod'] == 'chap'])
            target.update(dict(alias=data['alias']['iscsiAlias'], iqn=data['nodeName']['iscsiNodeName'], chap=chap))
            rc, data = self.request('storage-systems/%s/graph/xpath-filter?query=/sa/iscsiEntityData' % self.ssid)
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
                self.request('storage-systems/%s/iscsi/target-settings' % self.ssid, method='POST', data=body)
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
        if update and (not self.check_mode):
            try:
                self.request('storage-systems/%s/iscsi/entity' % self.ssid, method='POST', data=body)
            except Exception as err:
                self.module.fail_json(msg='Failed to update the iSCSI target settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def update(self):
        update = self.apply_iscsi_settings()
        update = self.apply_target_changes() or update
        target = self.target
        data = dict(((key, target[key]) for key in target if key in ['iqn', 'alias']))
        self.module.exit_json(msg='The interface settings have been updated.', changed=update, **data)