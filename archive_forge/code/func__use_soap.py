import logging
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2.client_base import Base
from saml2.client_base import LogoutError
from saml2.client_base import NoServiceDefined
from saml2.client_base import SignOnError
from saml2.httpbase import HTTPError
from saml2.ident import code
from saml2.ident import decode
from saml2.mdstore import locations
from saml2.s_utils import sid
from saml2.s_utils import status_message_factory
from saml2.s_utils import success_status_factory
from saml2.saml import AssertionIDRef
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.time_util import not_on_or_after
def _use_soap(self, destination, query_type, **kwargs):
    _create_func = getattr(self, f'create_{query_type}')
    _response_func = getattr(self, f'parse_{query_type}_response')
    try:
        response_args = kwargs['response_args']
        del kwargs['response_args']
    except KeyError:
        response_args = None
    qid, query = _create_func(destination, **kwargs)
    response = self.send_using_soap(query, destination)
    if response.status_code == 200:
        if not response_args:
            response_args = {'binding': BINDING_SOAP}
        else:
            response_args['binding'] = BINDING_SOAP
        logger.debug('Verifying response')
        if response_args:
            response = _response_func(response.content, **response_args)
        else:
            response = _response_func(response.content)
    else:
        raise HTTPError(f'{int(response.status_code)}:{response.error}')
    if response:
        logger.debug('OK response from %s', destination)
        return response
    else:
        logger.debug('NOT OK response from %s', destination)
    return None