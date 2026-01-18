import ipaddress
import time
from datetime import datetime
from enum import Enum
class SessionType(Enum):
    VERIFIED = 'verified'
    UNVERIFIED = 'unverified'