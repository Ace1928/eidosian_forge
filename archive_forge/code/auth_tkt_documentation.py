import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request

    Creates the `AuthTKTMiddleware
    <class-paste.auth.auth_tkt.AuthTKTMiddleware.html>`_.

    ``secret`` is required, but can be set globally or locally.
    