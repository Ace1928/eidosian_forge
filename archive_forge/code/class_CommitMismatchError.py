import boto.exception
from boto.compat import json
import requests
import boto
class CommitMismatchError(Exception):
    pass