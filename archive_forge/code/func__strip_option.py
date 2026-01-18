import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _strip_option(opts_str, opt_name, is_required=True, quotes_required=False, allow_multiple=False):
    opt_value = [] if allow_multiple else None
    opts_str = opts_str.strip().strip(',')
    if opt_name in opts_str:
        try:
            split_str = '%s=' % opt_name
            parts = opts_str.split(split_str)
            before = parts[0]
            after = parts[1]
            if len(parts) > 2:
                if allow_multiple:
                    after = split_str.join(parts[1:])
                    value, after = _strip_option(after, opt_name, is_required=is_required, quotes_required=quotes_required, allow_multiple=allow_multiple)
                    opt_value.extend(value)
                else:
                    raise exceptions.ValidationError(_("Option '%s' found more than once in argument --instance ") % opt_name + INSTANCE_METAVAR)
            quote = after[0]
            if quote in ["'", '"']:
                after = after[1:]
            else:
                if quotes_required:
                    raise exceptions.ValidationError(_("Invalid '%s' option. The value must be quoted. (Or perhaps you're missing quotes around the entire argument string)") % opt_name)
                quote = ''
            split_str = '%s,' % quote
            parts = after.split(split_str)
            value = str(parts[0]).strip()
            if allow_multiple:
                opt_value.append(value)
                opt_value = list(set(opt_value))
            else:
                opt_value = value
            opts_str = before + split_str.join(parts[1:])
        except IndexError:
            raise exceptions.ValidationError(_("Invalid '%(name)s' parameter. %(error)s.") % {'name': opt_name, 'error': INSTANCE_ERROR})
    if is_required and (not opt_value):
        raise exceptions.MissingArgs([opt_name], message=_("Missing option '%s' for argument --instance ") + INSTANCE_METAVAR)
    return (opt_value, opts_str.strip().strip(','))