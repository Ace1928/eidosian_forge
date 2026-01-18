from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException
class AuthStrategy:
    """
    This class represents one or more attempts to auth with an SSH server.

    By default, subclasses must at least accept an ``ssh_config``
    (`.SSHConfig`) keyword argument, but may opt to accept more as needed for
    their particular strategy.
    """

    def __init__(self, ssh_config):
        self.ssh_config = ssh_config
        self.log = get_logger(__name__)

    def get_sources(self):
        """
        Generator yielding `AuthSource` instances, in the order to try.

        This is the primary override point for subclasses: you figure out what
        sources you need, and ``yield`` them.

        Subclasses _of_ subclasses may find themselves wanting to do things
        like filtering or discarding around a call to `super`.
        """
        raise NotImplementedError

    def authenticate(self, transport):
        """
        Handles attempting `AuthSource` instances yielded from `get_sources`.

        You *normally* won't need to override this, but it's an option for
        advanced users.
        """
        succeeded = False
        overall_result = AuthResult(strategy=self)
        for source in self.get_sources():
            self.log.debug(f'Trying {source}')
            try:
                result = source.authenticate(transport)
                succeeded = True
            except Exception as e:
                result = e
                source_class = e.__class__.__name__
                self.log.info(f'Authentication via {source} failed with {source_class}')
            overall_result.append(SourceResult(source, result))
            if succeeded:
                break
        if not succeeded:
            raise AuthFailure(result=overall_result)
        return overall_result