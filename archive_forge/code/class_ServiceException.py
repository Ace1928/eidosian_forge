from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ServiceException(Exception):
    """Exception raised when a cloud storage provider request fails.

    This exception is raised only as a result of a failed remote call.
  """

    def __init__(self, reason, status=None, body=None):
        Exception.__init__(self)
        self.reason = reason
        self.status = status
        self.body = body

    def __repr__(self):
        return str(self)

    def __str__(self):
        message = '%s:' % self.__class__.__name__
        if self.status:
            message += ' %s' % self.status
        message += ' %s' % self.reason
        if self.body:
            message += '\n%s' % self.body
        return message