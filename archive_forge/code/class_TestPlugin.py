from zope.interface import provider
from twisted.plugin import IPlugin
from twisted.test.test_plugin import ITestPlugin, ITestPlugin2
@provider(ITestPlugin, IPlugin)
class TestPlugin:
    """
    A plugin used solely for testing purposes.
    """

    @staticmethod
    def test1() -> None:
        pass