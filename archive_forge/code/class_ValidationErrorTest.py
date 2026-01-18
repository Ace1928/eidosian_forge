import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class ValidationErrorTest(test_util.TestCase):

    def testStr_NoFieldName(self):
        """Test string version of ValidationError when no name provided."""
        self.assertEquals('Validation error', str(messages.ValidationError('Validation error')))

    def testStr_FieldName(self):
        """Test string version of ValidationError when no name provided."""
        validation_error = messages.ValidationError('Validation error')
        validation_error.field_name = 'a_field'
        self.assertEquals('Validation error', str(validation_error))