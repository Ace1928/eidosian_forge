import re
import argparse
import os
import fileinput
import logging
from xmlschema.cli import xsd_version_number, defuse_data
from xmlschema.validators import XMLSchema10, XMLSchema11
from ._observers import ObservedXMLSchema10, ObservedXMLSchema11

    Factory function for file based schema/validation cases.

    :param test_class_builder: the test class builder function.
    :param testfiles: a single or a list of testfiles indexes.
    :param suffix: the suffix ('xml' or 'xsd') to consider for cases.
    :param check_with_lxml: if `True` compare with lxml XMLSchema class,     reporting anomalies. Works only for XSD 1.0 tests.
    :param codegen: if `True` is provided checks code generation with XML data     bindings module for all tests. For default is `False` and code generation     is tested only for the cases where the same option is provided.
    :param verbosity: the unittest's verbosity, can be 0, 1 or 2.
    :return: a list of test classes.
    