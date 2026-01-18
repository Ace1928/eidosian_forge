from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, HAS_AZURE
class AzureRMRecordSet(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), relative_name=dict(type='str', required=True), zone_name=dict(type='str', required=True), record_type=dict(choices=RECORD_ARGSPECS.keys(), required=True, type='str'), record_mode=dict(choices=['append', 'purge'], default='purge'), state=dict(choices=['present', 'absent'], default='present', type='str'), time_to_live=dict(type='int', default=3600), records=dict(type='list', elements='dict'), metadata=dict(type='dict'), append_metadata=dict(type='bool', default=True))
        required_if = [('state', 'present', ['records'])]
        self.results = dict(changed=False)
        super(AzureRMRecordSet, self).__init__(self.module_arg_spec, required_if=required_if, supports_check_mode=True, skip_exec=True)
        record_subspec = RECORD_ARGSPECS.get(self.module.params['record_type'])
        self.module_arg_spec['records']['options'] = record_subspec
        self.resource_group = None
        self.relative_name = None
        self.zone_name = None
        self.record_type = None
        self.record_mode = None
        self.state = None
        self.time_to_live = None
        self.records = None
        self.metadata = None
        super(AzureRMRecordSet, self).__init__(self.module_arg_spec, required_if=required_if, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec.keys():
            setattr(self, key, kwargs[key])
        zone = self.dns_client.zones.get(self.resource_group, self.zone_name)
        if not zone:
            self.fail('The zone {0} does not exist in the resource group {1}'.format(self.zone_name, self.resource_group))
        try:
            self.log('Fetching Record Set {0}'.format(self.relative_name))
            record_set = self.dns_client.record_sets.get(self.resource_group, self.zone_name, self.relative_name, self.record_type)
            self.results['state'] = self.recordset_to_dict(record_set)
        except ResourceNotFoundError:
            record_set = None
        record_type_metadata = RECORDSET_VALUE_MAP.get(self.record_type)
        if self.state == 'present':
            self.input_sdk_records = self.create_sdk_records(self.records, self.record_type)
            if not record_set:
                changed = True
            else:
                server_records = getattr(record_set, record_type_metadata.get('attrname'))
                self.input_sdk_records, changed = self.records_changed(self.input_sdk_records, server_records)
                changed |= record_set.ttl != self.time_to_live
                old_metadata = self.results['state']['metadata'] if 'metadata' in self.results['state'] else dict()
                update_metadata, self.results['state']['metadata'] = self.update_metadata(old_metadata)
                if update_metadata:
                    changed = True
                self.metadata = self.results['state']['metadata']
            self.results['changed'] |= changed
        elif self.state == 'absent':
            if record_set:
                self.results['changed'] = True
        if self.check_mode:
            return self.results
        if self.results['changed']:
            if self.state == 'present':
                record_set_args = dict(ttl=self.time_to_live)
                record_set_args[record_type_metadata['attrname']] = self.input_sdk_records if record_type_metadata['is_list'] else self.input_sdk_records[0]
                record_set = self.dns_models.RecordSet(**record_set_args)
                if self.metadata:
                    record_set.metadata = self.metadata
                self.results['state'] = self.create_or_update(record_set)
            elif self.state == 'absent':
                self.delete_record_set()
        return self.results

    def create_or_update(self, record_set):
        try:
            record_set = self.dns_client.record_sets.create_or_update(resource_group_name=self.resource_group, zone_name=self.zone_name, relative_record_set_name=self.relative_name, record_type=self.record_type, parameters=record_set)
            return self.recordset_to_dict(record_set)
        except Exception as exc:
            self.fail('Error creating or updating dns record {0} - {1}'.format(self.relative_name, exc.message or str(exc)))

    def delete_record_set(self):
        try:
            self.dns_client.record_sets.delete(resource_group_name=self.resource_group, zone_name=self.zone_name, relative_record_set_name=self.relative_name, record_type=self.record_type)
        except Exception as exc:
            self.fail('Error deleting record set {0} - {1}'.format(self.relative_name, exc.message or str(exc)))
        return None

    def create_sdk_records(self, input_records, record_type):
        record = RECORDSET_VALUE_MAP.get(record_type)
        if not record:
            self.fail('record type {0} is not supported now'.format(record_type))
        record_sdk_class = getattr(self.dns_models, record.get('classobj'))
        return [record_sdk_class(**x) for x in input_records]

    def records_changed(self, input_records, server_records):
        if not isinstance(server_records, list):
            server_records = [server_records]
        input_set = set([self.module.jsonify(x.as_dict()) for x in input_records])
        server_set = set([self.module.jsonify(x.as_dict()) for x in server_records])
        if self.record_mode == 'append':
            input_set = server_set.union(input_set)
        changed = input_set != server_set
        records = [self.module.from_json(x) for x in input_set]
        return (self.create_sdk_records(records, self.record_type), changed)

    def recordset_to_dict(self, recordset):
        result = recordset.as_dict()
        result['type'] = result['type'].strip('Microsoft.Network/dnszones/')
        return result

    def update_metadata(self, metadata):
        metadata = metadata or dict()
        new_metadata = copy.copy(metadata) if isinstance(metadata, dict) else dict()
        param_metadata = self.metadata if isinstance(self.metadata, dict) else dict()
        append_metadata = self.append_metadata if self.metadata is not None else True
        changed = False
        for key, value in param_metadata.items():
            if not new_metadata.get(key) or new_metadata[key] != value:
                changed = True
                new_metadata[key] = value
        if not append_metadata:
            for key, value in metadata.items():
                if not param_metadata.get(key):
                    new_metadata.pop(key)
                    changed = True
        return (changed, new_metadata)