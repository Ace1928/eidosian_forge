import base64
import urllib
import requests
import requests.exceptions
from requests.adapters import HTTPAdapter, Retry
from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile
Helper function to split a range from 0 to total_length into bloksizes