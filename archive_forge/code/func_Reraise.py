from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import traceback
from googlecloudsdk.core.util import encoding
import six
def Reraise(self):
    six.reraise(type(self._exception), self._exception, self._traceback)