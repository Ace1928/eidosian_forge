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
class NewSSLContext(Setting):
    name = 'ssl_context'
    section = 'Server Hooks'
    validator = validate_callable(2)
    type = callable

    def ssl_context(config, default_ssl_context_factory):
        return default_ssl_context_factory()
    default = staticmethod(ssl_context)
    desc = '        Called when SSLContext is needed.\n\n        Allows customizing SSL context.\n\n        The callable needs to accept an instance variable for the Config and\n        a factory function that returns default SSLContext which is initialized\n        with certificates, private key, cert_reqs, and ciphers according to\n        config and can be further customized by the callable.\n        The callable needs to return SSLContext object.\n\n        Following example shows a configuration file that sets the minimum TLS version to 1.3:\n\n        .. code-block:: python\n\n            def ssl_context(conf, default_ssl_context_factory):\n                import ssl\n                context = default_ssl_context_factory()\n                context.minimum_version = ssl.TLSVersion.TLSv1_3\n                return context\n\n        .. versionadded:: 20.2\n        '