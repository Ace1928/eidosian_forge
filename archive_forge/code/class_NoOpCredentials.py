from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class NoOpCredentials(object):

    def __init__(self):
        pass

    def authorize(self, http_obj):
        return http_obj

    def set_store(self, store):
        pass