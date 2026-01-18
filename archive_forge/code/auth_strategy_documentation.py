from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException

        Handles attempting `AuthSource` instances yielded from `get_sources`.

        You *normally* won't need to override this, but it's an option for
        advanced users.
        