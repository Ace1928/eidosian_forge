import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
Used for loading in middleware (holdover from paste.deploy).