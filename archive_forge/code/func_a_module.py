from __future__ import (absolute_import, division, print_function)
from ansible.plugins.loader import action_loader, module_loader
def a_module(term):
    """
    Example:
      - 'community.general.ufw' is community.general.a_module
      - 'community.general.does_not_exist' is not community.general.a_module
    """
    try:
        for loader in (action_loader, module_loader):
            data = loader.find_plugin(term)
            if data is not None:
                return True
        return False
    except AnsiblePluginRemovedError:
        return False