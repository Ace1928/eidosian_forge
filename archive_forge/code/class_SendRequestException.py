from __future__ import absolute_import, division, print_function
class SendRequestException(Exception):

    def __init__(self, response, status_code):
        self.response = response
        self.status_code = status_code