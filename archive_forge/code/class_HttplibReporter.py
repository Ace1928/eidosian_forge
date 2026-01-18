from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import pickle
import random
import string
import sys
class HttplibReporter(object):

    def __init__(self):
        pass

    def request(self, endpoint, method=None, body=None, headers=None):
        https_con = http_client.HTTPSConnection(endpoint[8:].split('/')[0])
        https_con.request(method, endpoint, body=body, headers=headers)
        response = https_con.getresponse()
        return ({'status': response.status},)