from __future__ import absolute_import, division, print_function
import re
import importlib
import importlib.metadata
def _version_satisfied(version, requirements):
    """
    Checks whether version matches given version requirements.

    :param version: a string, in input form to _parse_version()
    :param requirements: as string, in input form to _parse_version_requirements()
    :returns: True if 'version' matches 'requirements'; False otherwise
    """
    version = _parse_version(version)
    requirements = _parse_version_requirements(requirements)
    for req in requirements:
        if not _compare_version(req[0], version, req[1]):
            return False
    return True