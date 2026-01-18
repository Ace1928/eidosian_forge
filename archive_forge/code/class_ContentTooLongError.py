import boto.exception
from boto.compat import json
import requests
import boto
class ContentTooLongError(Exception):
    """
    Content sent for Cloud Search indexing was too long

    This will usually happen when documents queued for indexing add up to more
    than the limit allowed per upload batch (5MB)

    """
    pass