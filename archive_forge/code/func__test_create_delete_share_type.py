import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
def _test_create_delete_share_type(self, microversion, is_public, dhss, spec_snapshot_support, spec_create_share_from_snapshot, spec_revert_to_snapshot_support, spec_mount_snapshot_support, extra_specs, description=None):
    share_type_name = data_utils.rand_name('manilaclient_functional_test')
    if extra_specs is None:
        extra_specs = {}
    share_type = self.create_share_type(name=share_type_name, driver_handles_share_servers=dhss, snapshot_support=spec_snapshot_support, create_share_from_snapshot=spec_create_share_from_snapshot, revert_to_snapshot=spec_revert_to_snapshot_support, mount_snapshot=spec_mount_snapshot_support, is_public=is_public, microversion=microversion, extra_specs=extra_specs, description=description)
    for key in self.create_keys:
        self.assertIn(key, share_type)
    self.assertEqual(share_type_name, share_type['Name'])
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.41'):
        self.assertEqual(description, share_type['Description'])
    else:
        self.assertNotIn('description', share_type)
    dhss_expected = 'driver_handles_share_servers : %s' % dhss
    self.assertEqual(dhss_expected, share_type['required_extra_specs'])
    expected_extra_specs = []
    for key, val in extra_specs.items():
        expected_extra_specs.append('{} : {}'.format(key, val).strip())
    if api_versions.APIVersion(microversion) < api_versions.APIVersion('2.24'):
        if 'snapshot_support' not in extra_specs:
            if spec_snapshot_support is None:
                expected_extra_specs.append('{} : {}'.format('snapshot_support', True).strip())
            else:
                expected_extra_specs.append('{} : {}'.format('snapshot_support', spec_snapshot_support).strip())
    elif spec_snapshot_support is not None:
        expected_extra_specs.append('{} : {}'.format('snapshot_support', spec_snapshot_support).strip())
    if spec_create_share_from_snapshot is not None:
        expected_extra_specs.append('{} : {}'.format('create_share_from_snapshot_support', spec_create_share_from_snapshot).strip())
    if spec_revert_to_snapshot_support is not None:
        expected_extra_specs.append('{} : {}'.format('revert_to_snapshot_support', spec_revert_to_snapshot_support).strip())
    if spec_mount_snapshot_support is not None:
        expected_extra_specs.append('{} : {}'.format('mount_snapshot_support', spec_mount_snapshot_support).strip())
    optional_extra_specs = share_type['optional_extra_specs']
    if optional_extra_specs == '':
        optional_extra_specs = []
    elif not isinstance(optional_extra_specs, list):
        optional_extra_specs = [optional_extra_specs]
    self.assertEqual(len(expected_extra_specs), len(optional_extra_specs))
    for e in optional_extra_specs:
        self.assertIn(e.strip(), expected_extra_specs)
    self.assertEqual('public' if is_public else 'private', share_type['Visibility'].lower())
    self.assertEqual('-', share_type['is_default'])
    st_id = share_type['ID']
    self._verify_access(share_type_id=st_id, is_public=is_public, microversion=microversion)
    self.admin_client.delete_share_type(st_id, microversion=microversion)
    self.admin_client.wait_for_share_type_deletion(st_id, microversion=microversion)
    share_types = self.admin_client.list_share_types(list_all=False, microversion=microversion)
    self.assertFalse(any((st_id == st['ID'] for st in share_types)))