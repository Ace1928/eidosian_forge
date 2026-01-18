import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
class WADLError(Exception):
    """An exception having to do with the state of the WADL application."""
    pass