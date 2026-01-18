from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def _test_image_ownership(self, headers, method):
    self.set_policy_rules({'get_image': '', 'add_image': '', 'publicize_image': '', 'communitize_image': '', 'add_member': ''})
    for visibility in ('community', 'public', 'shared'):
        path = '/v2/images'
        data = {'name': '%s-image' % visibility, 'visibility': visibility}
        response = self.api_post(path, json=data)
        image = response.json
        self.assertEqual(201, response.status_code)
        self.assertEqual(visibility, image['visibility'])
        if visibility == 'shared':
            path = '/v2/images/%s/members' % image['id']
            data = {'member': 'fake-project-id'}
            response = self.api_post(path, json=data)
            self.assertEqual(200, response.status_code)
        path = '/v2/images/%s/tags/Test_Tag_2' % image['id']
        response = self.api_request(method, path, headers=headers)
        self.assertEqual(403, response.status_code)