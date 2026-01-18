import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
Merges environment files into the stack input parameters.

    If a list of environment files have been specified, this call will
    pull the contents of each from the files dict, parse them as
    environments, and merge them into the stack input params. This
    behavior is the same as earlier versions of the Heat client that
    performed this params population client-side.

    :param environment_files: ordered names of the environment files
           found in the files dict
    :type  environment_files: list or None
    :param files: mapping of stack filenames to contents
    :type  files: dict
    :param params: parameters describing the stack
    :type  params: dict
    :param param_schemata: parameter schema dict
    :type  param_schemata: dict
    