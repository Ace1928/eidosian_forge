from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
class _SectionProxy(_Section):
    """Contains the properties for the 'proxy' section."""

    def __init__(self):
        super(_SectionProxy, self).__init__('proxy')
        self.address = self._Add('address', help_text='Hostname or IP address of proxy server.')
        self.port = self._Add('port', help_text='Port to use when connected to the proxy server.')
        self.rdns = self._Add('rdns', default=True, help_text='If True, DNS queries will not be performed locally, and instead, handed to the proxy to resolve. This is default behavior.')
        self.username = self._Add('username', help_text='Username to use when connecting, if the proxy requires authentication.')
        self.password = self._Add('password', help_text='Password to use when connecting, if the proxy requires authentication.')
        valid_proxy_types = sorted(http_proxy_types.PROXY_TYPE_MAP.keys())

        def ProxyTypeValidator(proxy_type):
            if proxy_type is not None and proxy_type not in valid_proxy_types:
                raise InvalidValueError('The proxy type property value [{0}] is not valid. Possible values: [{1}].'.format(proxy_type, ', '.join(valid_proxy_types)))
        self.proxy_type = self._Add('type', help_text='Type of proxy being used.  Supported proxy types are: [{0}].'.format(', '.join(valid_proxy_types)), validator=ProxyTypeValidator, choices=valid_proxy_types)
        self.use_urllib3_via_shim = self._AddBool('use_urllib3_via_shim', default=False, hidden=True, help_text='If True, use `urllib3` to make requests via `httplib2shim`.')