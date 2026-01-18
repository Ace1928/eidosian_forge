import collections
import importlib
import os
import pkgutil
def _append_config_options(imported_modules, config_options):
    for module in imported_modules:
        configs = module.list_opts()
        for key, val in configs.items():
            config_options[key].extend(val)