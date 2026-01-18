import time
import email
import socket
import logging
import functools
import collections
import pyzor.digest
import pyzor.account
import pyzor.message
import pyzor.hacks.py26
class ClientRunner(object):

    def __init__(self, routine):
        self.log = logging.getLogger('pyzor')
        self.routine = routine
        self.all_ok = True
        self.results = []

    def run(self, server, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        message = '%s:%s\t' % server
        response = None
        try:
            response = self.routine(*args, **kwargs)
            self.handle_response(response, message)
        except (pyzor.CommError, KeyError, ValueError) as e:
            self.results.append('%s%s\n' % (message, (e.code, str(e))))
            self.log.error('%s\t%s: %s', server, e.__class__.__name__, e)
            self.all_ok = False

    def handle_response(self, response, message):
        """mesaage is a string we've built up so far"""
        if not response.is_ok():
            self.all_ok = False
        self.results.append('%s%s\n' % (message, response.head_tuple()))