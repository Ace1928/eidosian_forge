from __future__ import (absolute_import, division, print_function)
from ._import_helper import HTTPError as _HTTPError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import raise_from
class MissingRequirementException(DockerException):

    def __init__(self, msg, requirement, import_exception):
        self.msg = msg
        self.requirement = requirement
        self.import_exception = import_exception

    def __str__(self):
        return self.msg