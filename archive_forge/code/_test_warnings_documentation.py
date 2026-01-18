import sys
import unittest
import warnings

This module has a number of tests that raise different kinds of warnings.
When the tests are run, the warnings are caught and their messages are printed
to stdout.  This module also accepts an arg that is then passed to
unittest.main to affect the behavior of warnings.
Test_TextTestRunner.test_warnings executes this script with different
combinations of warnings args and -W flags and check that the output is correct.
See #10535.
