import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
class AbiquoNodeDriverTest(TestCaseMixin, unittest.TestCase):
    """
    Abiquo Node Driver test suite
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the driver with the main user
        """
        AbiquoNodeDriver.connectionCls.conn_class = AbiquoMockHttp
        cls.driver = AbiquoNodeDriver('son', 'goku', 'http://dummy.host.com/api')

    def test_unauthorized_controlled(self):
        """
        Test the Unauthorized Exception is Controlled.

        Test, through the 'login' method, that a '401 Unauthorized'
        raises a 'InvalidCredsError' instead of the 'MalformedUrlException'
        """
        self.assertRaises(InvalidCredsError, AbiquoNodeDriver, 'son', 'goten', 'http://dummy.host.com/api')

    def test_forbidden_controlled(self):
        """
        Test the Forbidden Exception is Controlled.

        Test, through the 'list_images' method, that a '403 Forbidden'
        raises an 'ForbidenError' instead of the 'MalformedUrlException'
        """
        AbiquoNodeDriver.connectionCls.conn_class = AbiquoMockHttp
        conn = AbiquoNodeDriver('son', 'gohan', 'http://dummy.host.com/api')
        self.assertRaises(ForbiddenError, conn.list_images)

    def test_handle_other_errors_such_as_not_found(self):
        """
        Test common 'logical' exceptions are controlled.

        Test that common exception (normally 404-Not Found and 409-Conflict),
        that return an XMLResponse with the explanation of the errors are
        controlled.
        """
        self.driver = AbiquoNodeDriver('go', 'trunks', 'http://dummy.host.com/api')
        self.assertRaises(LibcloudError, self.driver.list_images)

    def test_ex_create_and_delete_empty_group(self):
        """
        Test the creation and deletion of an empty group.
        """
        group = self.driver.ex_create_group('libcloud_test_group')
        group.destroy()

    def test_create_node_no_image_raise_exception(self):
        """
        Test 'create_node' without image.

        Test the 'create_node' function without 'image' parameter raises
        an Exception
        """
        self.assertRaises(TypeError, self.driver.create_node)

    def test_list_locations_response(self):
        if not self.should_list_locations:
            return None
        locations = self.driver.list_locations()
        self.assertTrue(isinstance(locations, list))

    def test_create_node_specify_location(self):
        """
        Test you can create a node specifying the location.
        """
        image = self.driver.list_images()[0]
        location = self.driver.list_locations()[0]
        self.driver.create_node(image=image, location=location)

    def test_create_node_specify_wrong_location(self):
        """
        Test you can not create a node with wrong location.
        """
        image = self.driver.list_images()[0]
        location = NodeLocation(435, 'fake-location', 'Spain', self.driver)
        self.assertRaises(LibcloudError, self.driver.create_node, image=image, location=location)

    def test_create_node_specify_wrong_image(self):
        """
        Test image compatibility.

        Some locations only can handle a group of images, not all of them.
        Test you can not create a node with incompatible image-location.
        """
        image = NodeImage(3234, 'dummy-image', self.driver)
        location = self.driver.list_locations()[0]
        self.assertRaises(LibcloudError, self.driver.create_node, image=image, location=location)

    def test_create_node_specify_group_name(self):
        """
        Test 'create_node' into a concrete group.
        """
        image = self.driver.list_images()[0]
        self.driver.create_node(image=image, ex_group_name='new_group_name')

    def test_create_group_location_does_not_exist(self):
        """
        Test 'create_node' with an unexistent location.

        Defines a 'fake' location and tries to create a node into it.
        """
        location = NodeLocation(435, 'fake-location', 'Spain', self.driver)
        self.assertRaises(LibcloudError, self.driver.ex_create_group, name='new_group_name', location=location)

    def test_destroy_node_response(self):
        """
        'destroy_node' basic test.

        Override the destroy to return a different node available
        to be undeployed. (by default it returns an already undeployed node,
        for test creation).
        """
        self.driver = AbiquoNodeDriver('go', 'trunks', 'http://dummy.host.com/api')
        node = self.driver.list_nodes()[0]
        ret = self.driver.destroy_node(node)
        self.assertTrue(ret)

    def test_destroy_node_response_failed(self):
        """
        'destroy_node' asynchronous error.

        Test that the driver handles correctly when, for some reason,
        the 'destroy' job fails.
        """
        self.driver = AbiquoNodeDriver('muten', 'roshi', 'http://dummy.host.com/api')
        node = self.driver.list_nodes()[0]
        ret = self.driver.destroy_node(node)
        self.assertFalse(ret)

    def test_destroy_node_allocation_state(self):
        """
        Test the 'destroy_node' invalid state.

        Try to destroy a node when the node is not running.
        """
        self.driver = AbiquoNodeDriver('ve', 'geta', 'http://dummy.host.com/api')
        node = self.driver.list_nodes()[0]
        self.assertRaises(LibcloudError, self.driver.destroy_node, node)

    def test_destroy_not_deployed_group(self):
        """
        Test 'ex_destroy_group' when group is not deployed.
        """
        location = self.driver.list_locations()[0]
        group = self.driver.ex_list_groups(location)[1]
        self.assertTrue(group.destroy())

    def test_destroy_deployed_group(self):
        """
        Test 'ex_destroy_group' when there are machines running.
        """
        location = self.driver.list_locations()[0]
        group = self.driver.ex_list_groups(location)[0]
        self.assertTrue(group.destroy())

    def test_destroy_deployed_group_failed(self):
        """
        Test 'ex_destroy_group' fails.

        Test driver handles correctly when, for some reason, the
        asynchronous job fails.
        """
        self.driver = AbiquoNodeDriver('muten', 'roshi', 'http://dummy.host.com/api')
        location = self.driver.list_locations()[0]
        group = self.driver.ex_list_groups(location)[0]
        self.assertFalse(group.destroy())

    def test_destroy_group_invalid_state(self):
        """
        Test 'ex_destroy_group' invalid state.

        Test the Driver raises an exception when the group is in
        invalid temporal state.
        """
        self.driver = AbiquoNodeDriver('ve', 'geta', 'http://dummy.host.com/api')
        location = self.driver.list_locations()[0]
        group = self.driver.ex_list_groups(location)[1]
        self.assertRaises(LibcloudError, group.destroy)

    def test_run_node(self):
        """
        Test 'ex_run_node' feature.
        """
        node = self.driver.list_nodes()[0]
        self.driver.ex_run_node(node)

    def test_run_node_invalid_state(self):
        """
        Test 'ex_run_node' invalid state.

        Test the Driver raises an exception when try to run a
        node that is in invalid state to run.
        """
        self.driver = AbiquoNodeDriver('go', 'trunks', 'http://dummy.host.com/api')
        node = self.driver.list_nodes()[0]
        self.assertRaises(LibcloudError, self.driver.ex_run_node, node)

    def test_run_node_failed(self):
        """
        Test 'ex_run_node' fails.

        Test driver handles correctly when, for some reason, the
        asynchronous job fails.
        """
        self.driver = AbiquoNodeDriver('ten', 'shin', 'http://dummy.host.com/api')
        node = self.driver.list_nodes()[0]
        self.assertRaises(LibcloudError, self.driver.ex_run_node, node)

    def test_get_href(self):
        xml = '\n<datacenter>\n        <link href="http://10.60.12.7:80/api/admin/datacenters/2"\n        type="application/vnd.abiquo.datacenter+xml" rel="edit1"/>\n        <link href="http://10.60.12.7:80/ponies/bar/foo/api/admin/datacenters/3"\n        type="application/vnd.abiquo.datacenter+xml" rel="edit2"/>\n        <link href="http://vdcbridge.interoute.com:80/jclouds/apiouds/api/admin/enterprises/1234"\n        type="application/vnd.abiquo.datacenter+xml" rel="edit3"/>\n</datacenter>\n'
        elem = ET.XML(xml)
        href = get_href(element=elem, rel='edit1')
        self.assertEqual(href, '/admin/datacenters/2')
        href = get_href(element=elem, rel='edit2')
        self.assertEqual(href, '/admin/datacenters/3')
        href = get_href(element=elem, rel='edit3')
        self.assertEqual(href, '/admin/enterprises/1234')