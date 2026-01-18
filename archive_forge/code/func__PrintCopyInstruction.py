from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import select
import socket
import sys
import webbrowser
import wsgiref
from google_auth_oauthlib import flow as google_auth_flow
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as c_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import pkg_resources
from oauthlib.oauth2.rfc6749 import errors as rfc6749_errors
from requests import exceptions as requests_exceptions
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves.urllib import parse
def _PrintCopyInstruction(self, auth_response):
    con = console_attr.GetConsoleAttr()
    log.status.write(self._COPY_AUTH_RESPONSE_INSTRUCTION + ' ')
    log.status.Print(self._COPY_AUTH_RESPONSE_WARNING.format(bold=con.GetFontCode(bold=True), command=self._target_command, normal=con.GetFontCode()))
    log.status.write('\n')
    log.status.Print(auth_response)