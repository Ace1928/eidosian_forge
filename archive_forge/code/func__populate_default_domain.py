import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def _populate_default_domain(self):
    try:
        PROVIDERS.resource_api.get_domain(DEFAULT_DOMAIN_ID)
    except exception.DomainNotFound:
        root_domain = unit.new_domain_ref(id=resource_base.NULL_DOMAIN_ID, name=resource_base.NULL_DOMAIN_ID)
        PROVIDERS.resource_api.create_domain(resource_base.NULL_DOMAIN_ID, root_domain)
        domain = unit.new_domain_ref(description=u'The default domain', id=DEFAULT_DOMAIN_ID, name=u'Default')
        PROVIDERS.resource_api.create_domain(DEFAULT_DOMAIN_ID, domain)