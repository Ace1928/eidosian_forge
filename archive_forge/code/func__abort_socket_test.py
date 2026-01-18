import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
def _abort_socket_test(self, ex):
    try:
        self.loop.stop()
    finally:
        self.fail(ex)