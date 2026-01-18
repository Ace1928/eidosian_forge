from paste import httpexceptions
from paste import wsgilib
import flup.middleware.session

    Wraps the application in a session-managing middleware.
    The session service can then be found in
    ``environ['paste.flup_session_service']``
    