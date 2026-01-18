from __future__ import annotations
import json
import os
import sys
import textwrap
from collections import namedtuple
from datetime import datetime
from typing import Final, NoReturn
from uuid import uuid4
from streamlit import cli_util, env_util, file_util, util
from streamlit.logger import get_logger
def _verify_email(email: str) -> _Activation:
    """Verify the user's email address.

    The email can either be an empty string (if the user chooses not to enter
    it), or a string with a single '@' somewhere in it.

    Parameters
    ----------
    email : str

    Returns
    -------
    _Activation
        An _Activation object. Its 'is_valid' property will be True only if
        the email was validated.

    """
    email = email.strip()
    if len(email) > 0 and email.count('@') != 1:
        _LOGGER.error("That doesn't look like an email :(")
        return _Activation(None, False)
    return _Activation(email, True)