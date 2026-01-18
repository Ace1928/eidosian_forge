from __future__ import (absolute_import, division, print_function)
class ImporterAnsibleModule:
    """Replacement for AnsibleModule to support import testing."""

    def __init__(self, *args, **kwargs):
        raise ImporterAnsibleModuleException()