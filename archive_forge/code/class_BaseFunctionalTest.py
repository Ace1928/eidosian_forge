import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
class BaseFunctionalTest(base.TestCase):
    _wait_for_timeout_key = ''

    def setUp(self):
        super(BaseFunctionalTest, self).setUp()
        self.conn = connection.Connection(config=TEST_CLOUD_REGION)
        _disable_keep_alive(self.conn)
        self._demo_name = os.environ.get('OPENSTACKSDK_DEMO_CLOUD', 'devstack')
        self._demo_name_alt = os.environ.get('OPENSTACKSDK_DEMO_CLOUD_ALT', 'devstack-alt')
        self._op_name = os.environ.get('OPENSTACKSDK_OPERATOR_CLOUD', 'devstack-admin')
        self.config = openstack.config.OpenStackConfig()
        self._set_user_cloud()
        if self._op_name:
            self._set_operator_cloud()
        else:
            self.operator_cloud = None
        self.identity_version = self.user_cloud.config.get_api_version('identity')
        self.flavor = self._pick_flavor()
        self.image = self._pick_image()
        self._wait_for_timeout = int(os.getenv(self._wait_for_timeout_key, os.getenv('OPENSTACKSDK_FUNC_TEST_TIMEOUT', 300)))

    def _set_user_cloud(self, **kwargs):
        user_config = self.config.get_one(cloud=self._demo_name, **kwargs)
        self.user_cloud = connection.Connection(config=user_config)
        _disable_keep_alive(self.user_cloud)
        if self._demo_name_alt:
            user_config_alt = self.config.get_one(cloud=self._demo_name_alt, **kwargs)
            self.user_cloud_alt = connection.Connection(config=user_config_alt)
            _disable_keep_alive(self.user_cloud_alt)
        else:
            self.user_cloud_alt = None

    def _set_operator_cloud(self, **kwargs):
        operator_config = self.config.get_one(cloud=self._op_name, **kwargs)
        self.operator_cloud = connection.Connection(config=operator_config)
        _disable_keep_alive(self.operator_cloud)

    def _pick_flavor(self):
        """Pick a sensible flavor to run tests with.

        This returns None if the compute service is not present (e.g.
        ironic-only deployments).
        """
        if not self.user_cloud.has_service('compute'):
            return None
        flavors = self.user_cloud.list_flavors(get_extra=False)
        flavor_name = os.environ.get('OPENSTACKSDK_FLAVOR')
        if not flavor_name:
            flavor_name = _get_resource_value('flavor_name')
        if flavor_name:
            for flavor in flavors:
                if flavor.name == flavor_name:
                    return flavor
            raise self.failureException("Cloud does not have flavor '%s'", flavor_name)
        for flavor in sorted(flavors, key=operator.attrgetter('ram')):
            if 'performance' in flavor.name:
                return flavor
        for flavor in sorted(flavors, key=operator.attrgetter('ram')):
            if flavor.disk:
                return flavor
        raise self.failureException('No sensible flavor found')

    def _pick_image(self):
        """Pick a sensible image to run tests with.

        This returns None if the image service is not present.
        """
        if not self.user_cloud.has_service('image'):
            return None
        images = self.user_cloud.list_images()
        image_name = os.environ.get('OPENSTACKSDK_IMAGE')
        if not image_name:
            image_name = _get_resource_value('image_name')
        if image_name:
            for image in images:
                if image.name == image_name:
                    return image
            raise self.failureException("Cloud does not have image '%s'", image_name)
        for image in images:
            if image.name.startswith('cirros') and image.name.endswith('-uec'):
                return image
        for image in images:
            if image.name.startswith('cirros') and image.disk_format == 'qcow2':
                return image
        for image in images:
            if image.name.lower().startswith('ubuntu'):
                return image
        for image in images:
            if image.name.lower().startswith('centos'):
                return image
        raise self.failureException('No sensible image found')

    def addEmptyCleanup(self, func, *args, **kwargs):

        def cleanup():
            result = func(*args, **kwargs)
            self.assertIsNone(result)
        self.addCleanup(cleanup)

    def require_service(self, service_type, min_microversion=None, **kwargs):
        """Method to check whether a service exists

        Usage::

            class TestMeter(base.BaseFunctionalTest):
                def setUp(self):
                    super(TestMeter, self).setUp()
                    self.require_service('metering')

        :returns: True if the service exists, otherwise False.
        """
        if not self.conn.has_service(service_type):
            self.skipTest('Service {service_type} not found in cloud'.format(service_type=service_type))
        if not min_microversion:
            return
        data = self.conn.session.get_endpoint_data(service_type=service_type, **kwargs)
        if not (data.min_microversion and data.max_microversion and discover.version_between(data.min_microversion, data.max_microversion, min_microversion)):
            self.skipTest(f'Service {service_type} does not provide microversion {min_microversion}')

    def getUniqueString(self, prefix=None):
        """Generate unique resource name"""
        return (prefix if prefix else '') + '{time}-{uuid}'.format(time=int(time.time()), uuid=uuid.uuid4().hex)