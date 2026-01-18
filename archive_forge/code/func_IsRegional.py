from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def IsRegional(location):
    """Returns True if the location string is a GCP region."""
    return len(location.split('-')) == 2