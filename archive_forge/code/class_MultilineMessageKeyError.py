import collections
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils
class MultilineMessageKeyError(KeyError):

    def __init__(self, message, original_key):
        super(MultilineMessageKeyError, self).__init__(original_key)
        self.__message = message

    def __str__(self):
        return self.__message