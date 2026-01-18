import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def __GetAndStream(self, request):
    self.__GetFile(request)
    self.__download.StreamInChunks()