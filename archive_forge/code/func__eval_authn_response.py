from io import StringIO
import logging
import platform
import shelve
import sys
import traceback
from urllib import parse
from paste.httpexceptions import HTTPInternalServerError
from paste.httpexceptions import HTTPNotImplemented
from paste.httpexceptions import HTTPRedirection
from paste.httpexceptions import HTTPSeeOther
from paste.request import construct_url
from paste.request import parse_dict_querystring
from repoze.who.interfaces import IAuthenticator
from repoze.who.interfaces import IChallenger
from repoze.who.interfaces import IIdentifier
from repoze.who.interfaces import IMetadataProvider
from zope.interface import implementer
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import ecp
from saml2 import element_to_extension_element
from saml2 import xmldsig as ds
from saml2.client import Saml2Client
from saml2.client_base import ECP_SERVICE
from saml2.client_base import MIME_PAOS
from saml2.config import config_factory
from saml2.extension.pefim import SPCertEnc
from saml2.httputil import SeeOther
from saml2.httputil import getpath
from saml2.ident import code
from saml2.ident import decode
from saml2.profile import paos
from saml2.s_utils import sid
from saml2.samlp import Extensions
def _eval_authn_response(self, environ, post, binding=BINDING_HTTP_POST):
    logger.info('Got AuthN response, checking..')
    logger.info('Outstanding: %s', self.outstanding_queries)
    try:
        try:
            authresp = self.saml_client.parse_authn_request_response(post['SAMLResponse'][0], binding, self.outstanding_queries, self.outstanding_certs)
        except Exception as excp:
            logger.exception(f'Exception: {excp}')
            raise
        session_info = authresp.session_info()
    except TypeError as excp:
        logger.exception(f'Exception: {excp}')
        return None
    if session_info['came_from']:
        logger.debug('came_from << %s', session_info['came_from'])
        try:
            path, query = session_info['came_from'].split('?')
            environ['PATH_INFO'] = path
            environ['QUERY_STRING'] = query
        except ValueError:
            environ['PATH_INFO'] = session_info['came_from']
    logger.info('Session_info: %s', session_info)
    return session_info