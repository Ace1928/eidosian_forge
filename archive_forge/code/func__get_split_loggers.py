import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def _get_split_loggers(self, split_loggers):
    """Get a boolean value from the various argument sources.

        We default split_loggers to None in the kwargs of the Session
        constructor so we can track set vs. not set. We also accept
        split_loggers as a parameter in a few other places. In each place
        we want the parameter, if given by the user, to win.
        """
    if split_loggers is None:
        split_loggers = self._split_loggers
    if split_loggers is None:
        split_loggers = False
    return split_loggers