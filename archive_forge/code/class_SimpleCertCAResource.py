import flask_restful
from keystone.api._shared import json_home_relations
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class SimpleCertCAResource(flask_restful.Resource):

    @ks_flask.unenforced_api
    def get(self):
        raise exception.Gone(message=_('This API is no longer available due to the removal of support for PKI tokens.'))