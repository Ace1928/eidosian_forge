import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class ForwardedAllowIPS(Setting):
    name = 'forwarded_allow_ips'
    section = 'Server Mechanics'
    cli = ['--forwarded-allow-ips']
    meta = 'STRING'
    validator = validate_string_to_list
    default = os.environ.get('FORWARDED_ALLOW_IPS', '127.0.0.1')
    desc = '        Front-end\'s IPs from which allowed to handle set secure headers.\n        (comma separate).\n\n        Set to ``*`` to disable checking of Front-end IPs (useful for setups\n        where you don\'t know in advance the IP address of Front-end, but\n        you still trust the environment).\n\n        By default, the value of the ``FORWARDED_ALLOW_IPS`` environment\n        variable. If it is not defined, the default is ``"127.0.0.1"``.\n\n        .. note::\n\n            The interplay between the request headers, the value of ``forwarded_allow_ips``, and the value of\n            ``secure_scheme_headers`` is complex. Various scenarios are documented below to further elaborate.\n            In each case, we have a request from the remote address 134.213.44.18, and the default value of\n            ``secure_scheme_headers``:\n\n            .. code::\n\n                secure_scheme_headers = {\n                    \'X-FORWARDED-PROTOCOL\': \'ssl\',\n                    \'X-FORWARDED-PROTO\': \'https\',\n                    \'X-FORWARDED-SSL\': \'on\'\n                }\n\n\n            .. list-table::\n                :header-rows: 1\n                :align: center\n                :widths: auto\n\n                * - ``forwarded-allow-ips``\n                  - Secure Request Headers\n                  - Result\n                  - Explanation\n                * - .. code::\n\n                        ["127.0.0.1"]\n                  - .. code::\n\n                        X-Forwarded-Proto: https\n                  - .. code::\n\n                        wsgi.url_scheme = "http"\n                  - IP address was not allowed\n                * - .. code::\n\n                        "*"\n                  - <none>\n                  - .. code::\n\n                        wsgi.url_scheme = "http"\n                  - IP address allowed, but no secure headers provided\n                * - .. code::\n\n                        "*"\n                  - .. code::\n\n                        X-Forwarded-Proto: https\n                  - .. code::\n\n                        wsgi.url_scheme = "https"\n                  - IP address allowed, one request header matched\n                * - .. code::\n\n                        ["134.213.44.18"]\n                  - .. code::\n\n                        X-Forwarded-Ssl: on\n                        X-Forwarded-Proto: http\n                  - ``InvalidSchemeHeaders()`` raised\n                  - IP address allowed, but the two secure headers disagreed on if HTTPS was used\n\n\n        '