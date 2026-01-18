from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleLookupError
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
from ansible.release import __version__ as ansible_version
from ansible.template import Templar
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
Generate list.