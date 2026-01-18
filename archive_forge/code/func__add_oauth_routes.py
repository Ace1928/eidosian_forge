from __future__ import annotations
import hashlib
import os
import typing
import urllib.parse
import warnings
from dataclasses import dataclass, field
import fastapi
from fastapi.responses import RedirectResponse
from huggingface_hub import HfFolder, whoami
from .utils import get_space
def _add_oauth_routes(app: fastapi.FastAPI) -> None:
    """Add OAuth routes to the FastAPI app (login, callback handler and logout)."""
    try:
        from authlib.integrations.base_client.errors import MismatchingStateError
        from authlib.integrations.starlette_client import OAuth
    except ImportError as e:
        raise ImportError('Cannot initialize OAuth to due a missing library. Please run `pip install gradio[oauth]` or add `gradio[oauth]` to your requirements.txt file in order to install the required dependencies.') from e
    msg = "OAuth is required but {} environment variable is not set. Make sure you've enabled OAuth in your Space by setting `hf_oauth: true` in the Space metadata."
    if OAUTH_CLIENT_ID is None:
        raise ValueError(msg.format('OAUTH_CLIENT_ID'))
    if OAUTH_CLIENT_SECRET is None:
        raise ValueError(msg.format('OAUTH_CLIENT_SECRET'))
    if OAUTH_SCOPES is None:
        raise ValueError(msg.format('OAUTH_SCOPES'))
    if OPENID_PROVIDER_URL is None:
        raise ValueError(msg.format('OPENID_PROVIDER_URL'))
    oauth = OAuth()
    oauth.register(name='huggingface', client_id=OAUTH_CLIENT_ID, client_secret=OAUTH_CLIENT_SECRET, client_kwargs={'scope': OAUTH_SCOPES}, server_metadata_url=OPENID_PROVIDER_URL + '/.well-known/openid-configuration')

    @app.get('/login/huggingface')
    async def oauth_login(request: fastapi.Request):
        """Endpoint that redirects to HF OAuth page."""
        redirect_uri = _generate_redirect_uri(request)
        return await oauth.huggingface.authorize_redirect(request, redirect_uri)

    @app.get('/login/callback')
    async def oauth_redirect_callback(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that handles the OAuth callback."""
        try:
            oauth_info = await oauth.huggingface.authorize_access_token(request)
        except MismatchingStateError:
            login_uri = '/login/huggingface'
            if '_target_url' in request.query_params:
                login_uri += '?' + urllib.parse.urlencode({'_target_url': request.query_params['_target_url']})
            for key in list(request.session.keys()):
                if key.startswith('_state_huggingface'):
                    request.session.pop(key)
            return RedirectResponse(login_uri)
        request.session['oauth_info'] = oauth_info
        return _redirect_to_target(request)

    @app.get('/logout')
    async def oauth_logout(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that logs out the user (e.g. delete cookie session)."""
        request.session.pop('oauth_info', None)
        return _redirect_to_target(request)