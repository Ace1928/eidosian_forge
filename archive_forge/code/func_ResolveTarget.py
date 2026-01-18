import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def ResolveTarget(build_file, target, toolset):
    [parsed_build_file, target, parsed_toolset] = ParseQualifiedTarget(target)
    if parsed_build_file:
        if build_file:
            build_file = os.path.normpath(os.path.join(os.path.dirname(build_file), parsed_build_file))
            if not os.path.isabs(build_file):
                build_file = RelativePath(build_file, '.')
        else:
            build_file = parsed_build_file
    if parsed_toolset:
        toolset = parsed_toolset
    return [build_file, target, toolset]