import unittest
import re
import os
from textwrap import dedent
from xml.etree.ElementTree import Element, iselement
from xmlschema.exceptions import XMLSchemaValueError
from xmlschema.names import XSD_NAMESPACE, XSI_NAMESPACE, XSD_SCHEMA
from xmlschema.helpers import get_namespace
from xmlschema.resources import fetch_namespaces
from xmlschema.validators import XMLSchema10
from ._helpers import etree_elements_assert_equal

        Checks schema or validation errors, checking information completeness of the
        instances and those number against expected.

        :param path: the path of the test case.
        :param expected: the number of expected errors.
        