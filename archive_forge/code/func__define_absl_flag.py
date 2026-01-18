from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
def _define_absl_flag(self, flag_instance, suppress):
    """Defines a flag from the flag_instance."""
    flag_name = flag_instance.name
    short_name = flag_instance.short_name
    argument_names = ['--' + flag_name]
    if short_name:
        argument_names.insert(0, '-' + short_name)
    if suppress:
        helptext = argparse.SUPPRESS
    else:
        helptext = flag_instance.help.replace('%', '%%')
    if flag_instance.boolean:
        argument_names.append('--no' + flag_name)
        self.add_argument(*argument_names, action=_BooleanFlagAction, help=helptext, metavar=flag_instance.name.upper(), flag_instance=flag_instance)
    else:
        self.add_argument(*argument_names, action=_FlagAction, help=helptext, metavar=flag_instance.name.upper(), flag_instance=flag_instance)