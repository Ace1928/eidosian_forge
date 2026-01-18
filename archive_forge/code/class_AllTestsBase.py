import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import (
import os
from glob import glob
import logging
from prov.tests import examples
import prov.model as pm
import rdflib as rl
from rdflib.compare import graph_diff
from io import BytesIO, StringIO
class AllTestsBase(TestExamplesBase, TestStatementsBase2, TestQualifiedNamesBase, TestAttributesBase2):
    """This is a test to include all available tests."""
    pass