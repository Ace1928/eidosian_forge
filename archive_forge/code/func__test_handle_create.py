import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_handle_create(self, is_public=True, projects=None):
    value = mock.MagicMock()
    volume_type_id = '927202df-1afb-497f-8368-9c2d2f26e5db'
    value.id = volume_type_id
    self.volume_types.create.return_value = value
    tmpl = self.stack.t.t
    props = tmpl['resources']['my_volume_type']['properties'].copy()
    props['is_public'] = is_public
    if projects:
        props['projects'] = projects
        project = collections.namedtuple('Project', ['id'])
        stub_projects = [project(p) for p in projects]
        self.project_list.side_effect = [p for p in stub_projects]
    self.my_volume_type.t = self.my_volume_type.t.freeze(properties=props)
    self.my_volume_type.reparse()
    self.my_volume_type.handle_create()
    self.volume_types.create.assert_called_once_with(name='volumeBackend', is_public=is_public, description=None)
    value.set_keys.assert_called_once_with({'volume_backend_name': 'lvmdriver'})
    self.assertEqual(volume_type_id, self.my_volume_type.resource_id)
    if projects:
        calls = []
        for p in projects:
            calls.append(mock.call(volume_type_id, p))
        self.volume_type_access.add_project_access.assert_has_calls(calls)