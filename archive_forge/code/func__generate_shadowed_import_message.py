import sys
import os
def _generate_shadowed_import_message(self, found_at):
    msg = 'It was not possible to initialize the debugger due to a module name conflict.\n\ni.e.: the module "%(import_name)s" could not be imported because it is shadowed by:\n%(found_at)s\nPlease rename this file/folder so that the original module from the standard library can be imported.' % {'import_name': self.import_name, 'found_at': found_at[0]}
    return msg