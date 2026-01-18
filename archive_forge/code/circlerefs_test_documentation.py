import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
Raise AssertionError if the wrapped code creates garbage with cycles.