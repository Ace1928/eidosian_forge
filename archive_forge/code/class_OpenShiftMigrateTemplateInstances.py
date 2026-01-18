from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
class OpenShiftMigrateTemplateInstances(AnsibleOpenshiftModule):

    def __init__(self, **kwargs):
        super(OpenShiftMigrateTemplateInstances, self).__init__(**kwargs)

    def patch_template_instance(self, resource, templateinstance):
        result = None
        try:
            result = resource.status.patch(templateinstance)
        except Exception as exc:
            self.fail_json(msg='Failed to migrate TemplateInstance {0} due to: {1}'.format(templateinstance['metadata']['name'], to_native(exc)))
        return result.to_dict()

    @staticmethod
    def perform_migrations(templateinstances):
        ti_list = []
        ti_to_be_migrated = []
        ti_list = templateinstances.get('kind') == 'TemplateInstanceList' and templateinstances.get('items') or [templateinstances]
        for ti_elem in ti_list:
            objects = ti_elem['status'].get('objects')
            if objects:
                for i, obj in enumerate(objects):
                    object_type = obj['ref']['kind']
                    if object_type in transforms.keys() and obj['ref'].get('apiVersion') != transforms[object_type]:
                        ti_elem['status']['objects'][i]['ref']['apiVersion'] = transforms[object_type]
                        ti_to_be_migrated.append(ti_elem)
        return ti_to_be_migrated

    def execute_module(self):
        templateinstances = None
        namespace = self.params.get('namespace')
        results = {'changed': False, 'result': []}
        resource = self.find_resource('templateinstances', 'template.openshift.io/v1', fail=True)
        if namespace:
            try:
                templateinstances = resource.get(namespace=namespace).to_dict()
            except DynamicApiError as exc:
                self.fail_json(msg="Failed to retrieve TemplateInstances in namespace '{0}': {1}".format(namespace, exc.body), error=exc.status, status=exc.status, reason=exc.reason)
            except Exception as exc:
                self.fail_json(msg="Failed to retrieve TemplateInstances in namespace '{0}': {1}".format(namespace, to_native(exc)), error='', status='', reason='')
        else:
            templateinstances = resource.get().to_dict()
            ti_to_be_migrated = self.perform_migrations(templateinstances)
            if ti_to_be_migrated:
                if self.check_mode:
                    self.exit_json(**{'changed': True, 'result': ti_to_be_migrated})
                else:
                    for ti_elem in ti_to_be_migrated:
                        results['result'].append(self.patch_template_instance(resource, ti_elem))
                    results['changed'] = True
        self.exit_json(**results)