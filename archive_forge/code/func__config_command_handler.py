import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def _config_command_handler(self, args, screen_info=None):
    """Command handler for the "config" command."""
    del screen_info
    parsed = self._config_argparser.parse_args(args)
    if hasattr(parsed, 'property_name') and hasattr(parsed, 'property_value'):
        self._config.set(parsed.property_name, parsed.property_value)
        return self._config.summarize(highlight=parsed.property_name)
    else:
        return self._config.summarize()