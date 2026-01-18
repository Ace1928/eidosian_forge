from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwareStoragePolicyManager(SPBM):

    def __init__(self, module):
        super(VmwareStoragePolicyManager, self).__init__(module)
        self.rest_client = VmwareRestClient(module)

    def create_mob_tag_values(self, tags):
        return pbm.capability.types.DiscreteSet(values=tags)

    def create_mob_capability_property_instance(self, tag_id, tag_operator, tags):
        return pbm.capability.PropertyInstance(id=tag_id, operator=tag_operator, value=self.create_mob_tag_values(tags))

    def create_mob_capability_constraint_instance(self, tag_id, tag_operator, tags):
        return pbm.capability.ConstraintInstance(propertyInstance=[self.create_mob_capability_property_instance(tag_id, tag_operator, tags)])

    def create_mob_capability_metadata_uniqueid(self, tag_category):
        return pbm.capability.CapabilityMetadata.UniqueId(namespace='http://www.vmware.com/storage/tag', id=tag_category)

    def create_mob_capability_instance(self, tag_id, tag_operator, tags, tag_category):
        return pbm.capability.CapabilityInstance(id=self.create_mob_capability_metadata_uniqueid(tag_category), constraint=[self.create_mob_capability_constraint_instance(tag_id, tag_operator, tags)])

    def create_mob_capability_constraints_subprofile(self, tag_id, tag_operator, tags, tag_category):
        return pbm.profile.SubProfileCapabilityConstraints.SubProfile(name='Tag based placement', capability=[self.create_mob_capability_instance(tag_id, tag_operator, tags, tag_category)])

    def create_mob_capability_subprofile(self, tag_id, tag_operator, tags, tag_category):
        return pbm.profile.SubProfileCapabilityConstraints(subProfiles=[self.create_mob_capability_constraints_subprofile(tag_id, tag_operator, tags, tag_category)])

    def create_mob_pbm_update_spec(self, tag_id, tag_operator, tags, tag_category, description):
        return pbm.profile.CapabilityBasedProfileUpdateSpec(description=description, constraints=self.create_mob_capability_subprofile(tag_id, tag_operator, tags, tag_category))

    def create_mob_pbm_create_spec(self, tag_id, tag_operator, tags, tag_category, description, name):
        return pbm.profile.CapabilityBasedProfileCreateSpec(name=name, description=description, resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), category='REQUIREMENT', constraints=self.create_mob_capability_subprofile(tag_id, tag_operator, tags, tag_category))

    def get_tag_constraints(self, capabilities):
        """
        Return tag constraints for a profile given its capabilities
        """
        tag_constraints = {}
        for capability in capabilities:
            for constraint in capability.constraint:
                if hasattr(constraint, 'propertyInstance'):
                    for propertyInstance in constraint.propertyInstance:
                        if hasattr(propertyInstance.value, 'values'):
                            tag_constraints['id'] = propertyInstance.id
                            tag_constraints['values'] = propertyInstance.value.values
                            tag_constraints['operator'] = propertyInstance.operator
        return tag_constraints

    def get_profile_manager(self):
        self.get_spbm_connection()
        return self.spbm_content.profileManager

    def get_storage_policies(self, profile_manager):
        profile_ids = profile_manager.PbmQueryProfile(resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), profileCategory='REQUIREMENT')
        profiles = []
        if profile_ids:
            profiles = profile_manager.PbmRetrieveContent(profileIds=profile_ids)
        return profiles

    def format_profile(self, profile):
        formatted_profile = {'name': profile.name, 'id': profile.profileId.uniqueId, 'description': profile.description}
        return formatted_profile

    def format_tag_mob_id(self, tag_category):
        return 'com.vmware.storage.tag.' + tag_category + '.property'

    def format_results_and_exit(self, results, policy, changed):
        results['vmware_vm_storage_policy'] = self.format_profile(policy)
        results['changed'] = changed
        self.module.exit_json(**results)

    def update_storage_policy(self, policy, pbm_client, results):
        expected_description = self.params.get('description')
        expected_tags = [self.params.get('tag_name')]
        expected_tag_category = self.params.get('tag_category')
        expected_tag_id = self.format_tag_mob_id(expected_tag_category)
        expected_operator = 'NOT'
        if self.params.get('tag_affinity'):
            expected_operator = None
        needs_change = False
        if policy.description != expected_description:
            needs_change = True
        if hasattr(policy.constraints, 'subProfiles'):
            for subprofile in policy.constraints.subProfiles:
                tag_constraints = self.get_tag_constraints(subprofile.capability)
                if tag_constraints['id'] == expected_tag_id:
                    if tag_constraints['values'] != expected_tags:
                        needs_change = True
                else:
                    needs_change = True
                if tag_constraints['operator'] != expected_operator:
                    needs_change = True
        else:
            needs_change = True
        if needs_change:
            pbm_client.PbmUpdate(profileId=policy.profileId, updateSpec=self.create_mob_pbm_update_spec(expected_tag_id, expected_operator, expected_tags, expected_tag_category, expected_description))
        self.format_results_and_exit(results, policy, needs_change)

    def remove_storage_policy(self, policy, pbm_client, results):
        pbm_client.PbmDelete(profileId=[policy.profileId])
        self.format_results_and_exit(results, policy, True)

    def create_storage_policy(self, policy, pbm_client, results):
        profile_ids = pbm_client.PbmCreate(createSpec=self.create_mob_pbm_create_spec(self.format_tag_mob_id(self.params.get('tag_category')), None, [self.params.get('tag_name')], self.params.get('tag_category'), self.params.get('description'), self.params.get('name')))
        policy = pbm_client.PbmRetrieveContent(profileIds=[profile_ids])
        self.format_results_and_exit(results, policy[0], True)

    def ensure_state(self):
        client = self.get_profile_manager()
        policies = self.get_storage_policies(client)
        policy_name = self.params.get('name')
        results = dict(changed=False, vmware_vm_storage_policy={})
        if self.params.get('state') == 'present':
            if self.params.get('tag_category') is None:
                self.module.fail_json(msg="tag_category is required when 'state' is 'present'")
            if self.params.get('tag_name') is None:
                self.module.fail_json(msg="tag_name is required when 'state' is 'present'")
            category_result = self.rest_client.get_category_by_name(self.params.get('tag_category'))
            if category_result is None:
                self.module.fail_json(msg='%s is not found in vCenter Server tag categories' % self.params.get('tag_category'))
            tag_result = self.rest_client.get_tag_by_category_name(self.params.get('tag_name'), self.params.get('tag_category'))
            if tag_result is None:
                self.module.fail_json(msg='%s is not found in vCenter Server tags' % self.params.get('tag_name'))
            for policy in policies:
                if policy.name == policy_name:
                    self.update_storage_policy(policy, client, results)
            self.create_storage_policy(policy, client, results)
        if self.params.get('state') == 'absent':
            for policy in policies:
                if policy.name == policy_name:
                    self.remove_storage_policy(policy, client, results)
            self.module.exit_json(**results)