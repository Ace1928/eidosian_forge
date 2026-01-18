import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
class ProfileApp(Application):
    name = u'ipython profile'
    description = profile_help
    examples = _main_examples
    subcommands = Dict(dict(create=(ProfileCreate, ProfileCreate.description.splitlines()[0]), list=(ProfileList, ProfileList.description.splitlines()[0]), locate=(ProfileLocate, ProfileLocate.description.splitlines()[0])))

    def start(self):
        if self.subapp is None:
            print('No subcommand specified. Must specify one of: %s' % self.subcommands.keys())
            print()
            self.print_description()
            self.print_subcommands()
            self.exit(1)
        else:
            return self.subapp.start()