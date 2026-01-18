from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import pickle
import sys
import threading
import time
from googlecloudsdk.core import exceptions
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import queue   # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def ToPickleableResult(self):
    """Return a pickleable version of this _Result.

    Traceback objects can't be pickled, so we just pass through the exc_value.
    Also, some values and exceptions can't be pickled.

    Returns:
      _Result: a pickleable version of this result.
    """
    if self.exc_info:
        pickleable_result = _Result(error=self.exc_info[1])
    else:
        pickleable_result = self
    try:
        pickle.dumps(pickleable_result)
    except pickle.PicklingError as err:
        return _Result(error=err)
    except Exception as err:
        return _Result(error=pickle.PicklingError("Couldn't pickle result [{0}]: {1}".format(pickleable_result, six.text_type(err))))
    return pickleable_result