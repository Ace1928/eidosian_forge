import gettext
import os
import re
import textwrap
import warnings
from . import declarative
@classmethod
def _initialize_docstring(cls):
    """
        This changes the class's docstring to include information
        about all the messages this validator uses.
        """
    doc = [textwrap.dedent(cls.__doc__ or '').rstrip(), '\n\n**Messages**\n\n']
    for name, default in sorted(cls._messages.items()):
        default = re.sub('(%\\(.*?\\)[rsifcx])', '``\\1``', default)
        doc.append('``' + name + '``:\n')
        doc.append('  ' + default + '\n\n')
    cls.__doc__ = ''.join(doc)