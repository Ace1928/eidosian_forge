import json
import urllib.parse as urllib_parse
def _parse_debug_options(opts):
    """Debug options are semicolon separated key=value pairs
    """
    options = {}
    if not opts:
        return options
    for opt in opts.split(';'):
        try:
            key, value = opt.split('=')
        except ValueError:
            continue
        try:
            options[key] = DEBUG_OPTIONS_PARSER[key](value)
        except KeyError:
            continue
    return options