import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
def _test_tag_method_help(self, meth, exp_normal, exp_enhanced):
    """Vet the help text of the options added by the tag filtering helpers.

        :param meth: One of the ``add_tag_*`` methods.
        :param exp_normal: Expected help output without ``enhance_help``.
        :param exp_enhanced: Expected output with ``enhance_help`` set to
            ``help_enhancer``
        """
    if sys.version_info >= (3, 10):
        options_name = 'options'
    else:
        options_name = 'optional arguments'
    parser = argparse.ArgumentParser()
    meth(parser, 'test')
    self.assertEqual(exp_normal % options_name, parser.format_help())
    parser = argparse.ArgumentParser()
    meth(parser, 'test', enhance_help=help_enhancer)
    self.assertEqual(exp_enhanced % options_name, parser.format_help())