from pkg_resources import resource_string
from testresources import TestResource
from wadllib.application import Application
from launchpadlib.testing.launchpad import FakeLaunchpad
class FakeLaunchpadResource(TestResource):

    def make(self, dependency_resources):
        return FakeLaunchpad(application=Application('https://api.example.com/testing/', resource_string('launchpadlib.testing', 'testing-wadl.xml')))