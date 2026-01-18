import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestAddProjectToImage(TestImage):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    _image = image_fakes.create_one_image()
    new_member = image_fakes.create_one_image_member(attrs={'image_id': _image.id, 'member_id': project.id})
    columns = ('created_at', 'image_id', 'member_id', 'schema', 'status', 'updated_at')
    datalist = (new_member.created_at, _image.id, new_member.member_id, new_member.schema, new_member.status, new_member.updated_at)

    def setUp(self):
        super().setUp()
        self.image_client.find_image.return_value = self._image
        self.image_client.add_member.return_value = self.new_member
        self.project_mock.get.return_value = self.project
        self.domain_mock.get.return_value = self.domain
        self.cmd = _image.AddProjectToImage(self.app, None)

    def test_add_project_to_image_no_option(self):
        arglist = [self._image.id, self.project.id]
        verifylist = [('image', self._image.id), ('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.add_member.assert_called_with(image=self._image.id, member_id=self.project.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_add_project_to_image_with_option(self):
        arglist = [self._image.id, self.project.id, '--project-domain', self.domain.id]
        verifylist = [('image', self._image.id), ('project', self.project.id), ('project_domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.add_member.assert_called_with(image=self._image.id, member_id=self.project.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)