import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class MissingDependencies(DependencyFailure):
    """Raised when a entity has dependencies that can not be satisfied.

    :param who: the entity that caused the missing dependency to be triggered.
    :param requirements: the dependency which were not satisfied.

    Further arguments are interpreted as for in
    :py:class:`~taskflow.exceptions.TaskFlowException`.
    """
    MESSAGE_TPL = "'%(who)s' requires %(requirements)s but no other entity produces said requirements"
    METHOD_TPL = "'%(method)s' method on "

    def __init__(self, who, requirements, cause=None, method=None):
        message = self.MESSAGE_TPL % {'who': who, 'requirements': requirements}
        if method:
            message = self.METHOD_TPL % {'method': method} + message
        super(MissingDependencies, self).__init__(message, cause=cause)
        self.missing_requirements = requirements