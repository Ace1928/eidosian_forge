import os
import urllib.parse
import uuid
from lxml import etree
from oslo_config import fixture as config
import requests
from keystoneclient.auth import conf
from keystoneclient.contrib.auth.v3 import saml2
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import saml2_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib.federation import saml as saml_manager
class _AuthentiatedResponseMissingTokenID(_AuthenticatedResponse):
    headers = {}