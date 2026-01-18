import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _dec_unres(target):
    return _decode_unreserved(target, normalize_case=True, encode_stray_percents=percents)