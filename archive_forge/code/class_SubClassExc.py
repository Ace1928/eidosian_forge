import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
class SubClassExc(exception.UnexpectedError):
    debug_message_format = 'Debug Message: %(debug_info)s'