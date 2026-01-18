import hashlib
import json
import os
from django import http
from django import shortcuts
from django.conf import settings
from django.core import urlresolvers
from django.shortcuts import redirect
from django.utils import html
import jsonpickle
from six.moves.urllib import parse
from oauth2client import client
from oauth2client.contrib import django_util
from oauth2client.contrib.django_util import get_storage
from oauth2client.contrib.django_util import signals
def _get_flow_for_token(csrf_token, request):
    """ Looks up the flow in session to recover information about requested
    scopes.

    Args:
        csrf_token: The token passed in the callback request that should
            match the one previously generated and stored in the request on the
            initial authorization view.

    Returns:
        The OAuth2 Flow object associated with this flow based on the
        CSRF token.
    """
    flow_pickle = request.session.get(_FLOW_KEY.format(csrf_token), None)
    return None if flow_pickle is None else jsonpickle.decode(flow_pickle)