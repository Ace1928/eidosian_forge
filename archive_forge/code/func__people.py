import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _people(dist: 'Distribution', val: List[dict], _root_dir: _Path, kind: str):
    field = []
    email_field = []
    for person in val:
        if 'name' not in person:
            email_field.append(person['email'])
        elif 'email' not in person:
            field.append(person['name'])
        else:
            addr = Address(display_name=person['name'], addr_spec=person['email'])
            email_field.append(str(addr))
    if field:
        _set_config(dist, kind, ', '.join(field))
    if email_field:
        _set_config(dist, f'{kind}_email', ', '.join(email_field))