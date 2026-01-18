import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def add_get_handler(self, key, handler):
    self.get_handlers[key] = handler