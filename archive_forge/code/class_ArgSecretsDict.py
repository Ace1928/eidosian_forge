from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
class ArgSecretsDict(arg_parsers.ArgDict):
    """ArgDict customized for holding secrets configuration."""

    def __init__(self, key_type=None, value_type=None, spec=None, min_length=0, max_length=None, allow_key_only=False, required_keys=None, operators=None):
        """Initializes the base ArgDict by forwarding the parameters."""
        super(ArgSecretsDict, self).__init__(key_type=key_type, value_type=value_type, spec=spec, min_length=min_length, max_length=max_length, allow_key_only=allow_key_only, required_keys=required_keys, operators=operators)

    def __call__(self, arg_value):
        secrets_dict = collections.OrderedDict(sorted(six.iteritems(super(ArgSecretsDict, self).__call__(arg_value))))
        _ValidateSecrets(secrets_dict)
        return secrets_dict