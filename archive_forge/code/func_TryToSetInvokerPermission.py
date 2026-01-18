from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import env_vars as env_vars_api_util
from googlecloudsdk.api_lib.functions.v1 import exceptions as function_exceptions
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as v2_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions.v1.deploy import enum_util
from googlecloudsdk.command_lib.functions.v1.deploy import labels_util
from googlecloudsdk.command_lib.functions.v1.deploy import source_util
from googlecloudsdk.command_lib.functions.v1.deploy import trigger_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import urllib
def TryToSetInvokerPermission():
    """Try to make the invoker permission be what we said it should.

    This is for executing in the polling loop, and will stop trying as soon as
    it succeeds at making a change.
    """
    if stop_trying_perm_set[0]:
        return
    try:
        if ensure_all_users_invoke:
            api_util.AddFunctionIamPolicyBinding(function.name)
            stop_trying_perm_set[0] = True
        elif deny_all_users_invoke:
            stop_trying_perm_set[0] = api_util.RemoveFunctionIamPolicyBindingIfFound(function.name)
    except calliope_exceptions.HttpException:
        stop_trying_perm_set[0] = True
        log.warning('Setting IAM policy failed, try `%s`' % _CreateBindPolicyCommand(function_ref))