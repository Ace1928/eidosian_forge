from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
class HostMixin(ParametersMixin):
    """
    Host Mixin to extend a :class:`ForemanAnsibleModule` (or any subclass) to work with host-related entities (Hosts, Hostgroups).

    This adds many optional parameters that are specific to Hosts and Hostgroups to the module.
    It also includes :class:`ParametersMixin`.
    """

    def __init__(self, **kwargs):
        foreman_spec = dict(compute_resource=dict(type='entity'), compute_profile=dict(type='entity'), domain=dict(type='entity'), subnet=dict(type='entity'), subnet6=dict(type='entity', resource_type='subnets'), root_pass=dict(no_log=True), realm=dict(type='entity'), architecture=dict(type='entity'), operatingsystem=dict(type='entity'), medium=dict(aliases=['media'], type='entity'), ptable=dict(type='entity'), pxe_loader=dict(choices=['PXELinux BIOS', 'PXELinux UEFI', 'Grub UEFI', 'Grub2 BIOS', 'Grub2 ELF', 'Grub2 UEFI', 'Grub2 UEFI SecureBoot', 'Grub2 UEFI HTTP', 'Grub2 UEFI HTTPS', 'Grub2 UEFI HTTPS SecureBoot', 'iPXE Embedded', 'iPXE UEFI HTTP', 'iPXE Chain BIOS', 'iPXE Chain UEFI', 'None']), environment=dict(type='entity'), puppetclasses=dict(type='entity_list', resolve=False), config_groups=dict(type='entity_list'), puppet_proxy=dict(type='entity', resource_type='smart_proxies'), puppet_ca_proxy=dict(type='entity', resource_type='smart_proxies'), openscap_proxy=dict(type='entity', resource_type='smart_proxies'), content_source=dict(type='entity', scope=['organization'], resource_type='smart_proxies'), lifecycle_environment=dict(type='entity', scope=['organization']), kickstart_repository=dict(type='entity', scope=['organization'], optional_scope=['lifecycle_environment', 'content_view'], resource_type='repositories'), content_view=dict(type='entity', scope=['organization'], optional_scope=['lifecycle_environment']), activation_keys=dict(no_log=False))
        foreman_spec.update(kwargs.pop('foreman_spec', {}))
        required_plugins = kwargs.pop('required_plugins', []) + [('katello', ['activation_keys', 'content_source', 'lifecycle_environment', 'kickstart_repository', 'content_view']), ('openscap', ['openscap_proxy'])]
        mutually_exclusive = kwargs.pop('mutually_exclusive', []) + [['medium', 'kickstart_repository']]
        super(HostMixin, self).__init__(foreman_spec=foreman_spec, required_plugins=required_plugins, mutually_exclusive=mutually_exclusive, **kwargs)

    def run(self, **kwargs):
        entity = self.lookup_entity('entity')
        if not self.desired_absent:
            if 'activation_keys' in self.foreman_params:
                if 'parameters' not in self.foreman_params:
                    parameters = [param for param in (entity or {}).get('parameters', []) if param['name'] != 'kt_activation_keys']
                else:
                    parameters = self.foreman_params['parameters']
                ak_param = {'name': 'kt_activation_keys', 'parameter_type': 'string', 'value': self.foreman_params.pop('activation_keys')}
                self.foreman_params['parameters'] = parameters + [ak_param]
            elif 'parameters' in self.foreman_params and entity is not None:
                current_ak_param = next((param for param in entity.get('parameters') if param['name'] == 'kt_activation_keys'), None)
                desired_ak_param = next((param for param in self.foreman_params['parameters'] if param['name'] == 'kt_activation_keys'), None)
                if current_ak_param and desired_ak_param is None:
                    self.foreman_params['parameters'].append(current_ak_param)
        self.validate_parameters()
        return super(HostMixin, self).run(**kwargs)