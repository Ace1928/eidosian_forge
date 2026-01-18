import boto.exception
from boto.compat import json
import requests
import boto
class SearchServiceException(Exception):
    pass