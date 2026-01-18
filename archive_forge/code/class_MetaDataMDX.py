import hashlib
from hashlib import sha1
import importlib
from itertools import chain
import json
import logging
import os
from os.path import isfile
from os.path import join
from re import compile as regex_compile
import sys
from warnings import warn as _warn
import requests
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import md
from saml2 import saml
from saml2 import samlp
from saml2 import xmldsig
from saml2 import xmlenc
from saml2.extension.algsupport import NAMESPACE as NS_ALGSUPPORT
from saml2.extension.algsupport import DigestMethod
from saml2.extension.algsupport import SigningMethod
from saml2.extension.idpdisc import BINDING_DISCO
from saml2.extension.idpdisc import DiscoveryResponse
from saml2.extension.mdattr import NAMESPACE as NS_MDATTR
from saml2.extension.mdattr import EntityAttributes
from saml2.extension.mdrpi import NAMESPACE as NS_MDRPI
from saml2.extension.mdrpi import RegistrationInfo
from saml2.extension.mdrpi import RegistrationPolicy
from saml2.extension.mdui import NAMESPACE as NS_MDUI
from saml2.extension.mdui import Description
from saml2.extension.mdui import DisplayName
from saml2.extension.mdui import InformationURL
from saml2.extension.mdui import Logo
from saml2.extension.mdui import PrivacyStatementURL
from saml2.extension.mdui import UIInfo
from saml2.extension.shibmd import NAMESPACE as NS_SHIBMD
from saml2.extension.shibmd import Scope
from saml2.httpbase import HTTPBase
from saml2.md import NAMESPACE as NS_MD
from saml2.md import ArtifactResolutionService
from saml2.md import EntitiesDescriptor
from saml2.md import EntityDescriptor
from saml2.md import NameIDMappingService
from saml2.md import SingleSignOnService
from saml2.mdie import to_dict
from saml2.s_utils import UnknownSystemEntity
from saml2.s_utils import UnsupportedBinding
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import split_len
from saml2.time_util import add_duration
from saml2.time_util import before
from saml2.time_util import instant
from saml2.time_util import str_to_time
from saml2.time_util import valid
from saml2.validate import NotValid
from saml2.validate import valid_instance
class MetaDataMDX(InMemoryMetaData):
    """
    Uses the MDQ protocol to fetch entity information.
    The protocol is defined at:
    https://datatracker.ietf.org/doc/draft-young-md-query-saml/
    """

    @staticmethod
    def sha1_entity_transform(entity_id):
        entity_id_sha1 = hashlib.sha1(entity_id.encode('utf-8')).hexdigest()
        transform = f'{{sha1}}{entity_id_sha1}'
        return transform

    def __init__(self, url=None, security=None, cert=None, entity_transform=None, freshness_period=None, http_client_timeout=None, **kwargs):
        """
        :params url: mdx service url
        :params security: SecurityContext()
        :params cert: certificate used to check signature of signed metadata
        :params entity_transform: function transforming (e.g. base64,
        sha1 hash or URL quote
        hash) the entity id. It is applied to the entity id before it is
        concatenated with the request URL sent to the MDX server. Defaults to
        sha1 transformation.
        :params freshness_period: a duration in the format described at
        https://www.w3.org/TR/xmlschema-2/#duration
        :params http_client_timeout: timeout of http requests
        """
        super().__init__(None, **kwargs)
        if not url:
            raise SAMLError('URL for MDQ server not specified.')
        self.url = url.rstrip('/')
        if entity_transform:
            self.entity_transform = entity_transform
        else:
            self.entity_transform = MetaDataMDX.sha1_entity_transform
        self.cert = cert
        self.security = security
        self.freshness_period = freshness_period or DEFAULT_FRESHNESS_PERIOD
        self.expiration_date = {}
        self.http_client_timeout = http_client_timeout
        self.node_name = f'{EntityDescriptor.c_namespace}:{EntityDescriptor.c_tag}'

    def load(self, *args, **kwargs):
        pass

    def _fetch_metadata(self, item):
        mdx_url = f'{self.url}/entities/{self.entity_transform(item)}'
        response = requests.get(mdx_url, headers={'Accept': SAML_METADATA_CONTENT_TYPE}, timeout=self.http_client_timeout)
        if response.status_code != 200:
            error_msg = f'Fething {item}: Got response status {response.status_code}'
            logger.warning(error_msg)
            raise KeyError(error_msg)
        _txt = response.content
        if not self.parse_and_check_signature(_txt):
            error_msg = f'Fething {item}: invalid signature'
            logger.error(error_msg)
            raise KeyError(error_msg)
        curr_time = str_to_time(instant())
        self.expiration_date[item] = add_duration(curr_time, self.freshness_period)
        return self.entity[item]

    def _is_metadata_fresh(self, item):
        return before(self.expiration_date[item])

    def __getitem__(self, item):
        if item not in self.entity:
            entity = self._fetch_metadata(item)
        elif not self._is_metadata_fresh(item):
            msg = f'Metadata for {item} have expired; refreshing metadata'
            logger.info(msg)
            _ = self.entity.pop(item)
            entity = self._fetch_metadata(item)
        else:
            entity = self.entity[item]
        return entity

    def single_sign_on_service(self, entity_id, binding=None, typ='idpsso'):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, 'idpsso_descriptor', 'single_sign_on_service', binding)