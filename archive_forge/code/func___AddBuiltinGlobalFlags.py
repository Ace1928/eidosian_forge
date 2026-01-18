from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def __AddBuiltinGlobalFlags(self, top_element):
    """Adds in calliope builtin global flags.

    This needs to happen immediately after the top group is loaded and before
    any other groups are loaded.  The flags must be present so when sub groups
    are loaded, the flags propagate down.

    Args:
      top_element: backend._CommandCommon, The root of the command tree.
    """
    calliope_base.FLAGS_FILE_FLAG.AddToParser(top_element.ai)
    calliope_base.FLATTEN_FLAG.AddToParser(top_element.ai)
    calliope_base.FORMAT_FLAG.AddToParser(top_element.ai)
    if self.__version_func is not None:
        top_element.ai.add_argument('-v', '--version', do_not_propagate=True, category=calliope_base.COMMONLY_USED_FLAGS, action=actions.FunctionExitAction(self.__version_func), help='Print version information and exit. This flag is only available at the global level.')
    top_element.ai.add_argument('--configuration', metavar='CONFIGURATION', category=calliope_base.COMMONLY_USED_FLAGS, help='        The configuration to use for this command invocation. For more\n        information on how to use configurations, run:\n        `gcloud topic configurations`.  You can also use the {0} environment\n        variable to set the equivalent of this flag for a terminal\n        session.'.format(config.CLOUDSDK_ACTIVE_CONFIG_NAME))
    top_element.ai.add_argument('--verbosity', choices=log.OrderedVerbosityNames(), default=log.DEFAULT_VERBOSITY_STRING, category=calliope_base.COMMONLY_USED_FLAGS, help='Override the default verbosity for this command.', action=actions.StoreProperty(properties.VALUES.core.verbosity))
    top_element.ai.add_argument('--user-output-enabled', metavar=' ', nargs='?', default=None, const='true', choices=('true', 'false'), action=actions.DeprecationAction('--user-output-enabled', warn='The `{flag_name}` flag will no longer support the explicit use of the `true/false` optional value in an upcoming release.', removed=False, show_message=lambda _: False, action=actions.StoreBooleanProperty(properties.VALUES.core.user_output_enabled)), help='Print user intended output to the console.')
    top_element.ai.add_argument('--log-http', default=None, action=actions.StoreBooleanProperty(properties.VALUES.core.log_http), help='Log all HTTP server requests and responses to stderr.')
    top_element.ai.add_argument('--authority-selector', default=None, action=actions.StoreProperty(properties.VALUES.auth.authority_selector), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
    top_element.ai.add_argument('--authorization-token-file', default=None, action=actions.StoreProperty(properties.VALUES.auth.authorization_token_file), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
    top_element.ai.add_argument('--credential-file-override', action=actions.StoreProperty(properties.VALUES.auth.credential_file_override), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
    top_element.ai.add_argument('--http-timeout', default=None, action=actions.StoreProperty(properties.VALUES.core.http_timeout), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
    FLAG_INTERNAL_FLAG_FILE_LINE.AddToParser(top_element.ai)