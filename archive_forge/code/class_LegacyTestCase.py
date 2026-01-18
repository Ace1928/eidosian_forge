from __future__ import annotations
import os
import sys
import unittest
import textwrap
from typing import Any
from . import markdown, Markdown, util
class LegacyTestCase(unittest.TestCase, metaclass=LegacyTestMeta):
    """
    A [`unittest.TestCase`][] subclass for running Markdown's legacy file-based tests.

    A subclass should define various properties which point to a directory of
    text-based test files and define various behaviors/defaults for those tests.
    The following properties are supported:

    Attributes:
        location (str): A path to the directory of test files. An absolute path is preferred.
        exclude (list[str]): A list of tests to exclude. Each test name should comprise the filename
            without an extension.
        normalize (bool): A boolean value indicating if the HTML should be normalized. Default: `False`.
        input_ext (str): A string containing the file extension of input files. Default: `.txt`.
        output_ext (str): A string containing the file extension of expected output files. Default: `html`.
        default_kwargs (Kwargs[str, Any]): The default set of keyword arguments for all test files in the directory.

    In addition, properties can be defined for each individual set of test files within
    the directory. The property should be given the name of the file without the file
    extension. Any spaces and dashes in the filename should be replaced with
    underscores. The value of the property should be a `Kwargs` instance which
    contains the keyword arguments that should be passed to `Markdown` for that
    test file. The keyword arguments will "update" the `default_kwargs`.

    When the class instance is created, it will walk the given directory and create
    a separate `Unitttest` for each set of test files using the naming scheme:
    `test_filename`. One `Unittest` will be run for each set of input and output files.
    """
    pass