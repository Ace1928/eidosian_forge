from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class BadRequestException(ServiceException):
    """Exception raised for malformed requests.

    Where it is possible to detect invalid arguments prior to sending them
    to the server, an ArgumentException should be raised instead.
  """