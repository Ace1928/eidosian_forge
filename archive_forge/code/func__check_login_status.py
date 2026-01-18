from __future__ import annotations
import json
import warnings
from typing import Literal
from gradio_client.documentation import document
from gradio.components import Button
from gradio.context import Context
from gradio.routes import Request
def _check_login_status(self, request: Request) -> LoginButton:
    session = getattr(request, 'session', None) or getattr(request.request, 'session', None)
    if session is None or 'oauth_info' not in session:
        return LoginButton(value=self.value, interactive=True)
    else:
        username = session['oauth_info']['userinfo']['preferred_username']
        logout_text = self.logout_value.format(username)
        return LoginButton(logout_text, interactive=True)