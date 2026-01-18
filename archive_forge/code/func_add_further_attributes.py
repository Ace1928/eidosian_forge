import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def add_further_attributes(record):
    record.add_attributes([(EX_NS['tag1'], 'hello'), (EX_NS['tag2'], 'bye'), (EX2_NS['tag3'], 'hi'), (EX_NS['tag1'], 'hello\nover\nmore\nlines')])