import logging
import optparse
import shutil
import sys
import textwrap
from contextlib import suppress
from typing import Any, Dict, Generator, List, Tuple
from pip._internal.cli.status_codes import UNKNOWN_ERROR
from pip._internal.configuration import Configuration, ConfigurationError
from pip._internal.utils.misc import redact_auth_from_url, strtobool
def _get_ordered_configuration_items(self) -> Generator[Tuple[str, Any], None, None]:
    override_order = ['global', self.name, ':env:']
    section_items: Dict[str, List[Tuple[str, Any]]] = {name: [] for name in override_order}
    for section_key, val in self.config.items():
        if not val:
            logger.debug("Ignoring configuration key '%s' as it's value is empty.", section_key)
            continue
        section, key = section_key.split('.', 1)
        if section in override_order:
            section_items[section].append((key, val))
    for section in override_order:
        for key, val in section_items[section]:
            yield (key, val)