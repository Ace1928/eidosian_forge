import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.resource import schema
from keystone.server import flask as ks_flask
class DomainConfigGroupResource(DomainConfigBase):
    """Provides config group routing functionality.

    This class leans on DomainConfigBase to provide the following APIs:

    GET/HEAD /v3/domains/{domain_id}/config/{group}
    PATCH /v3/domains/{domain_id}/config/{group}
    DELETE /v3/domains/{domain_id}/config/{group}
    """