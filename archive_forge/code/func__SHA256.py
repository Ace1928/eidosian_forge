import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import baseauthenticator
def _SHA256(self, string):
    """Helper method to perform SHA256."""
    md = hashlib.sha256()
    md.update(string.encode())
    return md.digest()