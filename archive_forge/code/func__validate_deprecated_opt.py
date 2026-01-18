import importlib.metadata
import logging
import re
import sys
import yaml
from oslo_config import cfg
from oslo_config import generator
def _validate_deprecated_opt(group, option, opt_data):
    if group not in opt_data['deprecated_options']:
        return False
    name_data = [o['name'] for o in opt_data['deprecated_options'][group]]
    name_data += [o.get('dest') for o in opt_data['deprecated_options'][group]]
    return option in name_data