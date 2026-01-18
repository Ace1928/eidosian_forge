import datetime
import io
import logging
import os
import re
import subprocess
import sys
import time
import unittest
import warnings
import contextlib
import portend
import pytest
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import text_or_bytes, HTTPSConnection, ntob
from cherrypy.lib import httputil
from cherrypy.lib import gctools
def _test_method_sorter(_, x, y):
    """Monkeypatch the test sorter to always run test_gc last in each suite."""
    if x == 'test_gc':
        return 1
    if y == 'test_gc':
        return -1
    if x > y:
        return 1
    if x < y:
        return -1
    return 0