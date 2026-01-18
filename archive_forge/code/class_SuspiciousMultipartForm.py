import operator
from django.utils.hashable import make_hashable
class SuspiciousMultipartForm(SuspiciousOperation):
    """Suspect MIME request in multipart form data"""
    pass