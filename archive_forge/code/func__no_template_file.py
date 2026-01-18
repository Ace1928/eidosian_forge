from unittest import mock
from heat.common import exception
from heat.engine import environment
from heat.engine import resource as res
from heat.engine import service
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def _no_template_file(self, function):
    env = environment.Environment()
    info = environment.ResourceInfo(env.registry, ['ResourceWithWrongRefOnFile'], 'not_existing.yaml')
    mock_iterable = mock.MagicMock(return_value=iter([info]))
    with mock.patch('heat.engine.environment.ResourceRegistry.iterable_by', new=mock_iterable):
        ex = self.assertRaises(exception.InvalidGlobalResource, function, self.ctx, type_name='ResourceWithWrongRefOnFile')
        msg = 'There was an error loading the definition of the global resource type ResourceWithWrongRefOnFile.'
        self.assertIn(msg, str(ex))