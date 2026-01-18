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
def _make_flow(request, scopes, return_url=None):
    """Creates a Web Server Flow

    Args:
        request: A Django request object.
        scopes: the request oauth2 scopes.
        return_url: The URL to return to after the flow is complete. Defaults
            to the path of the current request.

    Returns:
        An OAuth2 flow object that has been stored in the session.
    """
    csrf_token = hashlib.sha256(os.urandom(1024)).hexdigest()
    request.session[_CSRF_KEY] = csrf_token
    state = json.dumps({'csrf_token': csrf_token, 'return_url': return_url})
    flow = client.OAuth2WebServerFlow(client_id=django_util.oauth2_settings.client_id, client_secret=django_util.oauth2_settings.client_secret, scope=scopes, state=state, redirect_uri=request.build_absolute_uri(urlresolvers.reverse('google_oauth:callback')))
    flow_key = _FLOW_KEY.format(csrf_token)
    request.session[flow_key] = jsonpickle.encode(flow)
    return flow