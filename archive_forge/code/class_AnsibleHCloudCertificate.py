from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.certificates import BoundCertificate
class AnsibleHCloudCertificate(AnsibleHCloud):
    represent = 'hcloud_certificate'
    hcloud_certificate: BoundCertificate | None = None

    def _prepare_result(self):
        return {'id': to_native(self.hcloud_certificate.id), 'name': to_native(self.hcloud_certificate.name), 'type': to_native(self.hcloud_certificate.type), 'fingerprint': to_native(self.hcloud_certificate.fingerprint), 'certificate': to_native(self.hcloud_certificate.certificate), 'not_valid_before': to_native(self.hcloud_certificate.not_valid_before), 'not_valid_after': to_native(self.hcloud_certificate.not_valid_after), 'domain_names': [to_native(domain) for domain in self.hcloud_certificate.domain_names], 'labels': self.hcloud_certificate.labels}

    def _get_certificate(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_certificate = self.client.certificates.get_by_id(self.module.params.get('id'))
            elif self.module.params.get('name') is not None:
                self.hcloud_certificate = self.client.certificates.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_certificate(self):
        self.module.fail_on_missing_params(required_params=['name'])
        params = {'name': self.module.params.get('name'), 'labels': self.module.params.get('labels')}
        if self.module.params.get('type') == 'uploaded':
            self.module.fail_on_missing_params(required_params=['certificate', 'private_key'])
            params['certificate'] = self.module.params.get('certificate')
            params['private_key'] = self.module.params.get('private_key')
            if not self.module.check_mode:
                try:
                    self.client.certificates.create(**params)
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
        else:
            self.module.fail_on_missing_params(required_params=['domain_names'])
            params['domain_names'] = self.module.params.get('domain_names')
            if not self.module.check_mode:
                try:
                    resp = self.client.certificates.create_managed(**params)
                    resp.action.wait_until_finished(max_retries=1000)
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_certificate()

    def _update_certificate(self):
        try:
            name = self.module.params.get('name')
            if name is not None and self.hcloud_certificate.name != name:
                self.module.fail_on_missing_params(required_params=['id'])
                if not self.module.check_mode:
                    self.hcloud_certificate.update(name=name)
                self._mark_as_changed()
            labels = self.module.params.get('labels')
            if labels is not None and self.hcloud_certificate.labels != labels:
                if not self.module.check_mode:
                    self.hcloud_certificate.update(labels=labels)
                self._mark_as_changed()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
        self._get_certificate()

    def present_certificate(self):
        self._get_certificate()
        if self.hcloud_certificate is None:
            self._create_certificate()
        else:
            self._update_certificate()

    def delete_certificate(self):
        self._get_certificate()
        if self.hcloud_certificate is not None:
            if not self.module.check_mode:
                try:
                    self.client.certificates.delete(self.hcloud_certificate)
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_certificate = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, type={'choices': ['uploaded', 'managed'], 'default': 'uploaded'}, domain_names={'type': 'list', 'elements': 'str', 'default': []}, certificate={'type': 'str'}, private_key={'type': 'str', 'no_log': True}, labels={'type': 'dict'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], required_if=[['state', 'present', ['name']]], supports_check_mode=True)