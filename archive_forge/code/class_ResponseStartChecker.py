import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
class ResponseStartChecker(object):

    def __init__(self, start_response):
        self.start_response = start_response
        self.response_started = False

    def __call__(self, *args):
        self.response_started = True
        self.start_response(*args)