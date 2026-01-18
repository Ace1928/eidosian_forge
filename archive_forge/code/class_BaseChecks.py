import abc
class BaseChecks(object, metaclass=abc.ABCMeta):
    """Base class providing upgrade checks.

    Stadium projects which want to provide their own upgrade checks to
    neutron-status CLI tool should inherit from this class.

    Each check method have to accept neutron.cmd.status.Checker
    class as an argument because all checkes will be run in context of
    this class.
    """

    @abc.abstractmethod
    def get_checks(self):
        """Get tuple with check methods and check names to run."""
        pass