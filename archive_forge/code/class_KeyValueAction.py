import argparse
from osc_lib.i18n import _
class KeyValueAction(argparse.Action):
    """A custom action to parse arguments as key=value pairs

    Ensures that ``dest`` is a dict and values are strings.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, {})
        if '=' in values:
            values_list = values.split('=', 1)
            if '' == values_list[0]:
                msg = _('Property key must be specified: %s')
                raise argparse.ArgumentTypeError(msg % str(values))
            else:
                getattr(namespace, self.dest, {}).update([values_list])
        else:
            msg = _("Expected 'key=value' type, but got: %s")
            raise argparse.ArgumentTypeError(msg % str(values))