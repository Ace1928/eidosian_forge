from unittest import mock
import glance.common.scripts as scripts
from glance.common.scripts.image_import import main as image_import
import glance.tests.utils as test_utils
class TestScripts(test_utils.BaseTestCase):

    def setUp(self):
        super(TestScripts, self).setUp()

    def test_run_task(self):
        task_id = mock.ANY
        task_type = 'import'
        context = mock.ANY
        task_repo = mock.ANY
        image_repo = mock.ANY
        image_factory = mock.ANY
        with mock.patch.object(image_import, 'run') as mock_run:
            scripts.run_task(task_id, task_type, context, task_repo, image_repo, image_factory)
        mock_run.assert_called_once_with(task_id, context, task_repo, image_repo, image_factory)