import base64
import binascii
import json
import time
import logging
from logging.config import dictConfig
from logging.config import fileConfig
import os
import socket
import sys
import threading
import traceback
from gunicorn import util
def _get_gunicorn_handler(self, log):
    for h in log.handlers:
        if getattr(h, '_gunicorn', False):
            return h