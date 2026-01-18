from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.ansible.utils.plugins.plugin_utils.base.cli_parser import CliParserBase
@staticmethod
def _check_reqs():
    """Check the prerequisites are installed for pyats/genie

        :return dict: A dict with a list of errors
        """
    errors = []
    if not HAS_GENIE:
        errors.append(missing_required_lib('genie'))
    if not HAS_PYATS:
        errors.append(missing_required_lib('pyats'))
    return errors