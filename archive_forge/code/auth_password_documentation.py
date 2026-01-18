from keystoneauth1 import exceptions as keystone_exceptions
from keystoneauth1 import session
from webob import exc
from heat.common import config
from heat.common import context
Build headers that represent authenticated user from auth token.