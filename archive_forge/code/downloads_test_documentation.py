import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
Integration tests for uploading and downloading to GCS.

These tests exercise most of the corner cases for upload/download of
files in apitools, via GCS. There are no performance tests here yet.
