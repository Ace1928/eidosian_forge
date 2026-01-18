import ast
from hacking import core
import re
def _filter_imports(self, module_name, alias):
    """Keep lists of logging and i18n imports."""
    if module_name in self.LOG_MODULES:
        self.logger_module_names.append(alias.asname or alias.name)
    elif module_name in self.I18N_MODULES:
        self.i18n_names[alias.asname or alias.name] = alias.name