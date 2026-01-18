from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from six.moves import urllib
def GetRecognizer(self, resource):
    request = self._messages.SpeechProjectsLocationsRecognizersGetRequest(name=resource.RelativeName())
    return self._RecognizerServiceForLocation(location=resource.Parent().Name()).Get(request)