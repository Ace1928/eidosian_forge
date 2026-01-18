from pbr import pbr_json
from pbr.tests import base
class TestJsonContent(base.BaseTestCase):

    @mock.patch('pbr.git._run_git_functions', return_value=True)
    @mock.patch('pbr.git.get_git_short_sha', return_value='123456')
    @mock.patch('pbr.git.get_is_release', return_value=True)
    def test_content(self, mock_get_is, mock_get_git, mock_run):
        cmd = mock.Mock()
        pbr_json.write_pbr_json(cmd, 'basename', 'pbr.json')
        cmd.write_file.assert_called_once_with('pbr', 'pbr.json', '{"git_version": "123456", "is_release": true}')