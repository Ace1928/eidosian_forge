import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class CannedIndex:

    def __init__(self, prefix, search_result):
        self.prefix = prefix
        self.result = search_result

    def get_topics(self, topic):
        return self.result