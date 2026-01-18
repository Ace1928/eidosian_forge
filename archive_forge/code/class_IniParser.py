from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.configparser import ConfigParser
from ansible.module_utils.common.text.converters import to_native
class IniParser(ConfigParser):
    """ Implements a configparser which sets the correct optionxform """

    def __init__(self):
        super().__init__()
        self.optionxform = str