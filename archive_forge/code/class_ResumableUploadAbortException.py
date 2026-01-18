from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ResumableUploadAbortException(ServiceException):
    """Exception raised for resumable uploads that cannot be retried later."""