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
class UVTestCase(BaseTestCase):
    implementation = 'uvloop'

    def new_loop(self):
        return uvloop.new_event_loop()

    def new_policy(self):
        return uvloop.EventLoopPolicy()