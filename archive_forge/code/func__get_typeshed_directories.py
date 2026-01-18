import os
import re
from functools import wraps
from collections import namedtuple
from typing import Dict, Mapping, Tuple
from pathlib import Path
from jedi import settings
from jedi.file_io import FileIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.stub_value import TypingModuleWrapper, StubModuleValue
from jedi.inference.value import ModuleValue
def _get_typeshed_directories(version_info):
    check_version_list = ['2and3', '3']
    for base in ['stdlib', 'third_party']:
        base_path = TYPESHED_PATH.joinpath(base)
        base_list = os.listdir(base_path)
        for base_list_entry in base_list:
            match = re.match('(\\d+)\\.(\\d+)$', base_list_entry)
            if match is not None:
                if match.group(1) == '3' and int(match.group(2)) <= version_info.minor:
                    check_version_list.append(base_list_entry)
        for check_version in check_version_list:
            is_third_party = base != 'stdlib'
            yield PathInfo(str(base_path.joinpath(check_version)), is_third_party)