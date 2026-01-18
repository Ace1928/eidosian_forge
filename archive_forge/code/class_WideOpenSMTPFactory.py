import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
class WideOpenSMTPFactory(StubSMTPFactory):
    """A fake smtp server that implements login by accepting anybody."""

    def login(self, user, password):
        self._calls.append(('login', user, password))