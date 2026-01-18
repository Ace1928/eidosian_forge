import copy
import logging
import os
import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier
from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs
def _parse_req_string(req_as_string: str) -> Requirement:
    try:
        req = get_requirement(req_as_string)
    except InvalidRequirement:
        if os.path.sep in req_as_string:
            add_msg = 'It looks like a path.'
            add_msg += deduce_helpful_msg(req_as_string)
        elif '=' in req_as_string and (not any((op in req_as_string for op in operators))):
            add_msg = '= is not a valid operator. Did you mean == ?'
        else:
            add_msg = ''
        msg = with_source(f'Invalid requirement: {req_as_string!r}')
        if add_msg:
            msg += f'\nHint: {add_msg}'
        raise InstallationError(msg)
    else:
        for spec in req.specifier:
            spec_str = str(spec)
            if spec_str.endswith(']'):
                msg = f"Extras after version '{spec_str}'."
                raise InstallationError(msg)
    return req