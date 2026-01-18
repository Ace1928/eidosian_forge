import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class TestSplitLoader(loading.BaseLoader):

    def get_options(self):
        opts = super(TestSplitLoader, self).get_options()
        opts += [loading.Opt('a'), loading.Opt('b')]
        return opts

    def create_plugin(self, a=None, b=None, **kwargs):
        if a:
            return PluginA(a)
        if b:
            return PluginB(b)
        raise AssertionError('Expected A or B')