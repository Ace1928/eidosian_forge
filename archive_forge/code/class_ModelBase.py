import abc
import stevedore
import keystone.conf
from keystone.i18n import _
class ModelBase(object, metaclass=abc.ABCMeta):
    """Interface for a limit model driver."""
    NAME = None
    DESCRIPTION = None
    MAX_PROJECT_TREE_DEPTH = None

    def check_limit(self, limits):
        """Check the new creating or updating limits if satisfy the model.

        :param limits: A list of the limit references to be checked.
        :type limits: A list of the limits. Each limit is a dictionary
                      reference containing all limit attributes.

        :raises keystone.exception.InvalidLimit: If any of the input limits
            doesn't satisfy the limit model.

        """
        raise NotImplementedError()