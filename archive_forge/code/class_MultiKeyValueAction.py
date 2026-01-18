import argparse
from osc_lib.i18n import _
class MultiKeyValueAction(argparse.Action):
    """A custom action to parse arguments as key1=value1,key2=value2 pairs

    Ensure that ``dest`` is a list. The list will finally contain multiple
    dicts, with key=value pairs in them.

    NOTE: The arguments string should be a comma separated key-value pairs.
    And comma(',') and equal('=') may not be used in the key or value.
    """

    def __init__(self, option_strings, dest, nargs=None, required_keys=None, optional_keys=None, **kwargs):
        """Initialize the action object, and parse customized options

        Required keys and optional keys can be specified when initializing
        the action to enable the key validation. If none of them specified,
        the key validation will be skipped.

        :param required_keys: a list of required keys
        :param optional_keys: a list of optional keys
        """
        if nargs:
            msg = _("Parameter 'nargs' is not allowed, but got %s")
            raise ValueError(msg % nargs)
        super(MultiKeyValueAction, self).__init__(option_strings, dest, **kwargs)
        if required_keys and (not isinstance(required_keys, list)):
            msg = _("'required_keys' must be a list")
            raise TypeError(msg)
        self.required_keys = set(required_keys or [])
        if optional_keys and (not isinstance(optional_keys, list)):
            msg = _("'optional_keys' must be a list")
            raise TypeError(msg)
        self.optional_keys = set(optional_keys or [])

    def validate_keys(self, keys):
        """Validate the provided keys.

        :param keys: A list of keys to validate.
        """
        valid_keys = self.required_keys | self.optional_keys
        if valid_keys:
            invalid_keys = [k for k in keys if k not in valid_keys]
            if invalid_keys:
                msg = _('Invalid keys %(invalid_keys)s specified.\nValid keys are: %(valid_keys)s')
                raise argparse.ArgumentTypeError(msg % {'invalid_keys': ', '.join(invalid_keys), 'valid_keys': ', '.join(valid_keys)})
        if self.required_keys:
            missing_keys = [k for k in self.required_keys if k not in keys]
            if missing_keys:
                msg = _('Missing required keys %(missing_keys)s.\nRequired keys are: %(required_keys)s')
                raise argparse.ArgumentTypeError(msg % {'missing_keys': ', '.join(missing_keys), 'required_keys': ', '.join(self.required_keys)})

    def __call__(self, parser, namespace, values, metavar=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, [])
        params = {}
        for kv in values.split(','):
            if '=' in kv:
                kv_list = kv.split('=', 1)
                if '' == kv_list[0]:
                    msg = _('Each property key must be specified: %s')
                    raise argparse.ArgumentTypeError(msg % str(kv))
                else:
                    params.update([kv_list])
            else:
                msg = _("Expected comma separated 'key=value' pairs, but got: %s")
                raise argparse.ArgumentTypeError(msg % str(kv))
        self.validate_keys(list(params))
        getattr(namespace, self.dest, []).append(params)