import argparse
from osc_lib.i18n import _
class NonNegativeAction(argparse.Action):
    """A custom action to check whether the value is non-negative or not

    Ensures the value is >= 0.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if int(values) >= 0:
            setattr(namespace, self.dest, values)
        else:
            msg = _('%s expected a non-negative integer')
            raise argparse.ArgumentTypeError(msg % str(option_string))