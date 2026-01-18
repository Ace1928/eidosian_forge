import operator
from django.utils.hashable import make_hashable
class TooManyFilesSent(SuspiciousOperation):
    """
    The number of fields in a GET or POST request exceeded
    settings.DATA_UPLOAD_MAX_NUMBER_FILES.
    """
    pass