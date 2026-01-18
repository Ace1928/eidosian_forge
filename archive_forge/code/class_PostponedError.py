import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
class PostponedError(object):

    def __init__(self, errorMsg):
        self.__errorMsg = errorMsg

    def __getitem__(self, item):
        raise error.PyAsn1Error(self.__errorMsg)