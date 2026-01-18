from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAutomationRunbook(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), automation_account_name=dict(type='str', required=True), runbook_type=dict(type='str', choices=['Script', 'Graph', 'PowerShellWorkflow', 'PowerShell', 'GraphPowerShellWorkflow', 'GraphPowerShell']), description=dict(type='str'), location=dict(type='str'), log_activity_trace=dict(type='int'), log_progress=dict(type='bool'), publish=dict(type='bool'), log_verbose=dict(type='bool'), state=dict(type='str', choices=['present', 'absent'], default='present'))
        self.results = dict()
        self.resource_group = None
        self.name = None
        self.automation_account_name = None
        self.runbook_type = None
        self.description = None
        self.log_activity_trace = None
        self.log_progress = None
        self.log_verbose = None
        self.location = None
        self.publish = None
        super(AzureRMAutomationRunbook, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec) + ['tags']:
            setattr(self, key, kwargs[key])
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        runbook = self.get()
        changed = False
        if self.state == 'present':
            if runbook:
                update_parameter = dict()
                if self.tags is not None:
                    update_tags, tags = self.update_tags(runbook['tags'])
                    if update_tags:
                        changed = True
                        update_parameter['tags'] = tags
                if self.description is not None and self.description != runbook['description']:
                    changed = True
                    update_parameter['description'] = self.description
                if self.log_activity_trace is not None and self.log_activity_trace != runbook['log_activity_trace']:
                    changed = True
                    update_parameter['log_activity_trace'] = self.log_activity_trace
                if self.log_progress is not None and self.log_progress != runbook['log_progress']:
                    changed = True
                    update_parameter['log_progress'] = self.log_progress
                if self.log_verbose is not None and self.log_verbose != runbook['log_verbose']:
                    changed = True
                    update_parameter['log_verbose'] = self.log_verbose
                if self.location is not None and self.location != runbook['location']:
                    changed = True
                    self.fail('Parameter error (location): The parameters {0} cannot be update'.format(self.location))
                if self.runbook_type is not None and self.runbook_type != runbook['runbook_type']:
                    changed = True
                    self.fail('Parameter error (runbook_type): The parameters {0} cannot be update'.format(self.runbook_type))
                if changed:
                    if not self.check_mode:
                        if update_parameter.get('log_activity_trace'):
                            runbook['log_activity_trace'] = update_parameter.get('log_activity_trace')
                        paramters = self.automation_models.RunbookCreateOrUpdateParameters(location=runbook['location'] if update_parameter.get('location') else update_parameter.get('location'), log_verbose=runbook['log_verbose'] if update_parameter.get('log_verbose') else update_parameter.get('log_verbose'), runbook_type=runbook['runbook_type'] if update_parameter.get('runbook_type') else update_parameter.get('runbook_type'), description=runbook['description'] if update_parameter.get('description') else update_parameter.get('description'), log_activity_trace=runbook['log_activity_trace'], tags=runbook['tags'] if update_parameter.get('tags') else update_parameter.get('tags'), log_progress=runbook['log_progress'] if update_parameter.get('log_progress') else update_parameter.get('log_progress'))
                        runbook = self.update_runbook(update_parameter)
            else:
                paramters = self.automation_models.RunbookCreateOrUpdateParameters(location=self.location, log_verbose=self.log_verbose, runbook_type=self.runbook_type, description=self.description, log_activity_trace=self.log_activity_trace, tags=self.tags, log_progress=self.log_progress)
                changed = True
                if not self.check_mode:
                    runbook = self.create_or_update(paramters)
            if not self.check_mode:
                if self.publish and runbook['state'] != 'Published':
                    changed = True
                    self.publish_runbook()
        else:
            changed = True
            if not self.check_mode:
                runbook = self.delete_automation_runbook()
        self.results['changed'] = changed
        self.results['state'] = runbook
        return self.results

    def get(self):
        try:
            response = self.automation_client.runbook.get(self.resource_group, self.automation_account_name, self.name)
            return self.to_dict(response)
        except ResourceNotFoundError:
            pass

    def publish_runbook(self):
        response = None
        try:
            response = self.automation_client.runbook.begin_publish(self.resource_group, self.automation_account_name, self.name)
        except Exception as exc:
            self.fail('Error when updating automation account {0}: {1}'.format(self.name, exc.message))

    def update_runbook(self, parameters):
        try:
            response = self.automation_client.runbook.update(self.resource_group, self.automation_account_name, self.name, parameters)
            return self.to_dict(response)
        except Exception as exc:
            self.fail('Error when updating automation account {0}: {1}'.format(self.name, exc.message))

    def create_or_update(self, parameters):
        try:
            response = self.automation_client.runbook.create_or_update(self.resource_group, self.automation_account_name, self.name, parameters)
            return self.to_dict(response)
        except Exception as exc:
            self.fail('Error when creating automation account {0}: {1}'.format(self.name, exc.message))

    def delete_automation_runbook(self):
        try:
            return self.automation_client.runbook.delete(self.resource_group, self.automation_account_name, self.name)
        except Exception as exc:
            self.fail('Error when deleting automation account {0}: {1}'.format(self.name, exc.message))

    def to_dict(self, runbook):
        if not runbook:
            return None
        runbook_dict = dict(id=runbook.id, type=runbook.type, name=runbook.name, tags=runbook.tags, location=runbook.location, runbook_type=runbook.runbook_type, runbook_content_link=runbook.publish_content_link, state=runbook.state, log_verbose=runbook.log_verbose, log_progress=runbook.log_progress, log_activity_trace=runbook.log_activity_trace, job_count=runbook.job_count, output_types=runbook.output_types, last_modified_by=runbook.last_modified_by, last_modified_time=runbook.last_modified_time, creation_time=runbook.creation_time, description=runbook.description)
        return runbook_dict