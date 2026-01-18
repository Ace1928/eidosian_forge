import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def _get_module_list(self, modules):
    """Build a list of module ids."""
    module_list = []
    for module in modules:
        module_info = {'id': base.getid(module)}
        module_list.append(module_info)
    return module_list