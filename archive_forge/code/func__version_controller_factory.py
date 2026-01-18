import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def _version_controller_factory(self, conf):
    return VersionController()