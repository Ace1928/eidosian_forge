import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
Returns the platform and device information.