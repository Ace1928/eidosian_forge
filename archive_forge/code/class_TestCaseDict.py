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
class TestCaseDict(collections.UserDict):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __setitem__(self, key, value):
        if key in self.data:
            raise RuntimeError('duplicate test {}.{}'.format(self.name, key))
        super().__setitem__(key, value)