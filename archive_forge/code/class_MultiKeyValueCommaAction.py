import argparse
from osc_lib.i18n import _
class MultiKeyValueCommaAction(MultiKeyValueAction):
    """Custom action to parse arguments from a set of key=value pair

    Ensures that ``dest`` is a dict.
    Parses dict by separating comma separated string into individual values
    Ex. key1=val1,val2,key2=val3 => {"key1": "val1,val2", "key2": "val3"}
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Overwrite the __call__ function of MultiKeyValueAction

        This is done to handle scenarios where we may have comma seperated
        data as a single value.
        """
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, [])
        params = {}
        key = ''
        for kv in values.split(','):
            if '=' in kv:
                kv_list = kv.split('=', 1)
                if '' == kv_list[0]:
                    msg = _("A key must be specified before '=': %s")
                    raise argparse.ArgumentTypeError(msg % str(kv))
                else:
                    params.update([kv_list])
                key = kv_list[0]
            else:
                try:
                    params[key] = '%s,%s' % (params[key], kv)
                except KeyError:
                    msg = _('A key=value pair is required: %s')
                    raise argparse.ArgumentTypeError(msg % str(kv))
        self.validate_keys(list(params))
        getattr(namespace, self.dest, []).append(params)