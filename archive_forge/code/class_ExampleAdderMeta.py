import keyword
import os
import re
import subprocess
import sys
from taskflow import test
class ExampleAdderMeta(type):
    """Translates examples into test cases/methods."""

    def __new__(cls, name, parents, dct):

        def generate_test(example_name):

            def test_example(self):
                self._check_example(example_name)
            return test_example
        for example_name, safe_name in iter_examples():
            test_name = 'test_%s' % safe_name
            dct[test_name] = generate_test(example_name)
        return type.__new__(cls, name, parents, dct)