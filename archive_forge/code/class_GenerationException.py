import os
import re
import six
from six.moves import urllib
from routes import request_config
class GenerationException(RoutesException):
    """Tossed during URL generation exceptions"""