import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
def _has_qml_decorated_class(class_list: List) -> bool:
    """Check for QML-decorated classes in the moc json output."""
    for d in class_list:
        class_infos = d.get('classInfos')
        if class_infos:
            for e in class_infos:
                if 'QML' in e['name']:
                    return True
    return False