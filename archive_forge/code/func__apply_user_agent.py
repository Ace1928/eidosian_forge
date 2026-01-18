import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def _apply_user_agent(headers, user_agent):
    """Adds a user-agent to the headers.

    Args:
        headers: dict, request headers to add / modify user
                 agent within.
        user_agent: str, the user agent to add.

    Returns:
        dict, the original headers passed in, but modified if the
        user agent is not None.
    """
    if user_agent is not None:
        if 'user-agent' in headers:
            headers['user-agent'] = user_agent + ' ' + headers['user-agent']
        else:
            headers['user-agent'] = user_agent
    return headers