from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
class APIImagePolicy(APIPolicyBase):

    def setUp(self):
        super(APIImagePolicy, self).setUp()
        self.image = mock.MagicMock()
        self.policy = policy.ImageAPIPolicy(self.context, self.image, enforcer=self.enforcer)

    def test_enforce(self):
        self.assertRaises(webob.exc.HTTPNotFound, super(APIImagePolicy, self).test_enforce)

    @mock.patch('glance.api.policy._enforce_image_visibility')
    def test_enforce_visibility(self, mock_enf):
        self.policy._enforce_visibility('something')
        mock_enf.assert_called_once_with(self.enforcer, self.context, 'something', mock.ANY)
        mock_enf.side_effect = exception.Forbidden
        self.assertRaises(webob.exc.HTTPForbidden, self.policy._enforce_visibility, 'something')
        mock_enf.side_effect = exception.ImageNotFound
        self.assertRaises(exception.ImageNotFound, self.policy._enforce_visibility, 'something')

    def test_update_property(self):
        with mock.patch.object(self.policy, '_enforce') as mock_enf:
            self.policy.update_property('foo', None)
            mock_enf.assert_called_once_with('modify_image')
        with mock.patch.object(self.policy, '_enforce_visibility') as mock_enf:
            self.policy.update_property('visibility', 'foo')
            mock_enf.assert_called_once_with('foo')

    def test_update_locations(self):
        self.policy.update_locations()
        self.enforcer.enforce.assert_called_once_with(self.context, 'set_image_location', mock.ANY)

    def test_delete_locations(self):
        self.policy.delete_locations()
        self.enforcer.enforce.assert_called_once_with(self.context, 'delete_image_location', mock.ANY)

    def test_delete_locations_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = True
        self.context.owner = 'someuser'
        self.image.owner = 'someotheruser'
        self.policy.delete_locations()
        self.context.is_admin = False
        self.context.owner = 'someuser'
        self.image.owner = 'someuser'
        self.policy.delete_locations()
        self.image.owner = 'someotheruser'
        self.assertRaises(exception.Forbidden, self.policy.delete_locations)
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.delete_locations()
            m.assert_called_once_with(self.context, self.image)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.delete_locations()
            self.assertFalse(m.called)

    def test_get_image_location(self):
        self.policy.get_image_location()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_image_location', mock.ANY)

    def test_enforce_exception_behavior(self):
        with mock.patch.object(self.policy.enforcer, 'enforce') as mock_enf:
            self.policy.update_property('foo', None)
            self.assertTrue(mock_enf.called)
            mock_enf.reset_mock()
            mock_enf.side_effect = exception.Forbidden
            self.assertRaises(webob.exc.HTTPNotFound, self.policy.update_property, 'foo', None)
            mock_enf.assert_has_calls([mock.call(mock.ANY, 'modify_image', mock.ANY), mock.call(mock.ANY, 'get_image', mock.ANY)])
            mock_enf.reset_mock()
            mock_enf.side_effect = [exception.Forbidden, lambda *a: None]
            self.assertRaises(webob.exc.HTTPForbidden, self.policy.update_property, 'foo', None)
            mock_enf.assert_has_calls([mock.call(mock.ANY, 'modify_image', mock.ANY), mock.call(mock.ANY, 'get_image', mock.ANY)])

    def test_get_image(self):
        self.policy.get_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_image', mock.ANY)

    def test_get_images(self):
        self.policy.get_images()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_images', mock.ANY)

    def test_add_image(self):
        generic_target = {'project_id': self.context.project_id, 'owner': self.context.project_id, 'visibility': 'private'}
        self.policy = policy.ImageAPIPolicy(self.context, {}, enforcer=self.enforcer)
        self.policy.add_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_image', generic_target)

    def test_add_image_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = False
        self.policy = policy.ImageAPIPolicy(self.context, {'owner': 'else'}, enforcer=self.enforcer)
        self.assertRaises(exception.Forbidden, self.policy.add_image)
        with mock.patch('glance.api.v2.policy.check_admin_or_same_owner') as m:
            self.policy.add_image()
            m.assert_called_once_with(self.context, {'project_id': 'else', 'owner': 'else', 'visibility': 'private'})
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_admin_or_same_owner') as m:
            self.policy.add_image()
            m.assert_not_called()

    def test_add_image_translates_owner_failure(self):
        self.policy = policy.ImageAPIPolicy(self.context, {'owner': 'else'}, enforcer=self.enforcer)
        self.policy.add_image()
        self.enforcer.enforce.side_effect = exception.Duplicate
        self.assertRaises(exception.Duplicate, self.policy.add_image)
        self.enforcer.enforce.side_effect = webob.exc.HTTPForbidden('original')
        exc = self.assertRaises(webob.exc.HTTPForbidden, self.policy.add_image)
        self.assertIn('You are not permitted to create images owned by', str(exc))
        self.policy = policy.ImageAPIPolicy(self.context, {}, enforcer=self.enforcer)
        exc = self.assertRaises(webob.exc.HTTPForbidden, self.policy.add_image)
        self.assertIn('original', str(exc))

    def test_delete_image(self):
        self.policy.delete_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'delete_image', mock.ANY)

    def test_delete_image_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = True
        self.context.owner = 'someuser'
        self.image.owner = 'someotheruser'
        self.policy.delete_image()
        self.context.is_admin = False
        self.context.owner = 'someuser'
        self.image.owner = 'someuser'
        self.policy.delete_image()
        self.image.owner = 'someotheruser'
        self.assertRaises(exception.Forbidden, self.policy.delete_image)
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.delete_image()
            m.assert_called_once_with(self.context, self.image)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.delete_image()
            self.assertFalse(m.called)

    def test_upload_image(self):
        self.policy.upload_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'upload_image', mock.ANY)

    def test_upload_image_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = True
        self.context.owner = 'someuser'
        self.image.owner = 'someotheruser'
        self.policy.upload_image()
        self.context.is_admin = False
        self.context.owner = 'someuser'
        self.image.owner = 'someuser'
        self.policy.upload_image()
        self.image.owner = 'someotheruser'
        self.assertRaises(exception.Forbidden, self.policy.upload_image)
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.upload_image()
            m.assert_called_once_with(self.context, self.image)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.upload_image()
            self.assertFalse(m.called)

    def test_download_image(self):
        self.policy.download_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'download_image', mock.ANY)

    def test_modify_image(self):
        self.policy.modify_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'modify_image', mock.ANY)

    def test_modify_image_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = True
        self.context.owner = 'someuser'
        self.image.owner = 'someotheruser'
        self.policy.modify_image()
        self.context.is_admin = False
        self.context.owner = 'someuser'
        self.image.owner = 'someuser'
        self.policy.modify_image()
        self.image.owner = 'someotheruser'
        self.assertRaises(exception.Forbidden, self.policy.modify_image)
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.modify_image()
            m.assert_called_once_with(self.context, self.image)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.modify_image()
            self.assertFalse(m.called)

    def test_deactivate_image(self):
        self.policy.deactivate_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'deactivate', mock.ANY)

    def test_deactivate_image_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = True
        self.context.owner = 'someuser'
        self.image.owner = 'someotheruser'
        self.policy.deactivate_image()
        self.context.is_admin = False
        self.context.owner = 'someuser'
        self.image.owner = 'someuser'
        self.policy.delete_image()
        self.image.owner = 'someotheruser'
        self.assertRaises(exception.Forbidden, self.policy.deactivate_image)
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.deactivate_image()
            m.assert_called_once_with(self.context, self.image)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.deactivate_image()
            self.assertFalse(m.called)

    def test_reactivate_image(self):
        self.policy.reactivate_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'reactivate', mock.ANY)

    def test_reactivate_image_falls_back_to_legacy(self):
        self.config(enforce_new_defaults=False, group='oslo_policy')
        self.config(enforce_scope=False, group='oslo_policy')
        self.context.is_admin = True
        self.context.owner = 'someuser'
        self.image.owner = 'someotheruser'
        self.policy.reactivate_image()
        self.context.is_admin = False
        self.context.owner = 'someuser'
        self.image.owner = 'someuser'
        self.policy.delete_image()
        self.image.owner = 'someotheruser'
        self.assertRaises(exception.Forbidden, self.policy.reactivate_image)
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.reactivate_image()
            m.assert_called_once_with(self.context, self.image)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')
        with mock.patch('glance.api.v2.policy.check_is_image_mutable') as m:
            self.policy.reactivate_image()
            self.assertFalse(m.called)

    def test_copy_image(self):
        self.policy.copy_image()
        self.enforcer.enforce.assert_called_once_with(self.context, 'copy_image', mock.ANY)