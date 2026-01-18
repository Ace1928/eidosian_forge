import time
import hashlib
import pyzor
class Account(object):

    def __init__(self, username, salt, key):
        self.username = username
        self.salt = salt
        self.key = key