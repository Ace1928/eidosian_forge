import operator
import struct
from passlib.utils.compat import izip
from passlib.crypto.digest import pbkdf2_hmac
from passlib.crypto.scrypt._salsa import salsa20
special bmix() method optimized for ``r=1`` case