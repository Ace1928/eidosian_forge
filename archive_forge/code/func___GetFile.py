import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def __GetFile(self, request):
    response = self.__client.objects.Get(request, download=self.__download)
    self.assertIsNone(response, msg='Unexpected nonempty response for file download: %s' % response)