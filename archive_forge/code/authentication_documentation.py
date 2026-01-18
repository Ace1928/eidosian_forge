import flask
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.receipt import handlers as receipt_handlers
Authenticate user and issue a token.