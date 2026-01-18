from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
Heat Template Resource for Keystone Service Endpoint.

    Keystone endpoint is just the URL that can be used for accessing a service
    within OpenStack. Endpoint can be accessed by admin, by services or public,
    i.e. everyone can use this endpoint.
    