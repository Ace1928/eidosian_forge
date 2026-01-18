import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class cmd_WithSeeAlso(commands.Command):
    __doc__ = 'A sample command.'