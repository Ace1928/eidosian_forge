from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import os
import pkgutil
import re
import six
from subprocess import PIPE
from subprocess import Popen
import gslib.addlhelp
from gslib.command import Command
from gslib.command import OLD_ALIAS_MAP
import gslib.commands
from gslib.exception import CommandException
from gslib.help_provider import HelpProvider
from gslib.help_provider import MAX_HELP_NAME_LEN
from gslib.utils import constants
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.system_util import IsRunningInteractively
from gslib.utils.system_util import GetTermLines
from gslib.utils import text_util
def _LoadHelpMaps(self):
    """Returns tuple of help type and help name.

    help type is a dict with key: help type
                             value: list of HelpProviders
    help name is a dict with key: help command name or alias
                             value: HelpProvider

    Returns:
      (help type, help name)
    """
    for _, module_name, _ in pkgutil.iter_modules(gslib.commands.__path__):
        __import__('gslib.commands.%s' % module_name)
    for _, module_name, _ in pkgutil.iter_modules(gslib.addlhelp.__path__):
        __import__('gslib.addlhelp.%s' % module_name)
    help_type_map = {}
    help_name_map = {}
    for s in gslib.help_provider.ALL_HELP_TYPES:
        help_type_map[s] = []
    for help_prov in itertools.chain(HelpProvider.__subclasses__(), Command.__subclasses__()):
        if help_prov is Command:
            continue
        gslib.help_provider.SanityCheck(help_prov, help_name_map)
        help_name_map[help_prov.help_spec.help_name] = help_prov
        for help_name_aliases in help_prov.help_spec.help_name_aliases:
            help_name_map[help_name_aliases] = help_prov
        help_type_map[help_prov.help_spec.help_type].append(help_prov)
    return (help_type_map, help_name_map)