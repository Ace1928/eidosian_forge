from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class EnclosureInfoModule(OneViewModuleBase):
    argument_spec = dict(name=dict(type='str'), options=dict(type='list', elements='raw'), params=dict(type='dict'))

    def __init__(self):
        super(EnclosureInfoModule, self).__init__(additional_arg_spec=self.argument_spec, supports_check_mode=True)

    def execute_module(self):
        info = {}
        if self.module.params['name']:
            enclosures = self._get_by_name(self.module.params['name'])
            if self.options and enclosures:
                info = self._gather_optional_info(self.options, enclosures[0])
        else:
            enclosures = self.oneview_client.enclosures.get_all(**self.facts_params)
        info['enclosures'] = enclosures
        return dict(changed=False, **info)

    def _gather_optional_info(self, options, enclosure):
        enclosure_client = self.oneview_client.enclosures
        info = {}
        if options.get('script'):
            info['enclosure_script'] = enclosure_client.get_script(enclosure['uri'])
        if options.get('environmentalConfiguration'):
            env_config = enclosure_client.get_environmental_configuration(enclosure['uri'])
            info['enclosure_environmental_configuration'] = env_config
        if options.get('utilization'):
            info['enclosure_utilization'] = self._get_utilization(enclosure, options['utilization'])
        return info

    def _get_utilization(self, enclosure, params):
        fields = view = refresh = filter = ''
        if isinstance(params, dict):
            fields = params.get('fields')
            view = params.get('view')
            refresh = params.get('refresh')
            filter = params.get('filter')
        return self.oneview_client.enclosures.get_utilization(enclosure['uri'], fields=fields, filter=filter, refresh=refresh, view=view)

    def _get_by_name(self, name):
        return self.oneview_client.enclosures.get_by('name', name)