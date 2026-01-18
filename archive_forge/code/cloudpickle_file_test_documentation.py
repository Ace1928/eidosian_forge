import os
import shutil
import sys
import tempfile
import unittest
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
In Cloudpickle, expected behaviour when pickling an opened file
    is to send its contents over the wire and seek to the same position.