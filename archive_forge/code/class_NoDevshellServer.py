import datetime
import json
import os
import socket
from oauth2client import _helpers
from oauth2client import client
class NoDevshellServer(Error):
    """Error when no Developer Shell server can be contacted."""