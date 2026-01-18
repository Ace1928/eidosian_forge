from heat_integrationtests.functional import functional_base
def _get_by_resource_name(self, changes, name, action):
    filtered_l = [x for x in changes[action] if x['resource_name'] == name]
    self.assertEqual(1, len(filtered_l))
    return filtered_l[0]