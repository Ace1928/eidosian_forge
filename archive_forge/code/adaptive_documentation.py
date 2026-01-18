import logging
import math
import threading
from botocore.retries import bucket, standard, throttling
Tracks the rate at which a client is sending a request.