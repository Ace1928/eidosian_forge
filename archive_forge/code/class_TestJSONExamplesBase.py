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
class TestJSONExamplesBase(object):
    """This is the base class for testing support for all the examples provided
    in prov.tests.examples.
    It is not runnable and needs to be included in a subclass of
    RoundTripTestCase.
    """

    def test_all_examples(self):
        counter = 0
        for name, graph in examples.tests:
            if name in ['datatypes']:
                logger.info('%d. Skipping the %s example', counter, name)
                continue
            counter += 1
            logger.info('%d. Testing the %s example', counter, name)
            g = graph()
            self.do_tests(g)