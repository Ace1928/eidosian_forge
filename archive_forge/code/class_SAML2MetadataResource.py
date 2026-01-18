import flask
import flask_restful
import http.client
from oslo_serialization import jsonutils
from oslo_log import log
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import schema
from keystone.federation import utils
from keystone.server import flask as ks_flask
class SAML2MetadataResource(flask_restful.Resource):

    @ks_flask.unenforced_api
    def get(self):
        """Get SAML2 metadata.

        GET/HEAD /OS-FEDERATION/saml2/metadata
        """
        metadata_path = CONF.saml.idp_metadata_path
        try:
            with open(metadata_path, 'r') as metadata_handler:
                metadata = metadata_handler.read()
        except IOError as e:
            raise exception.MetadataFileError(reason=e)
        resp = flask.make_response(metadata, http.client.OK)
        resp.headers['Content-Type'] = 'text/xml'
        return resp