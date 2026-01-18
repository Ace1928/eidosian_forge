from io import StringIO
from ... import branch as _mod_branch
from ... import controldir, errors
from ... import forge as _mod_forge
from ... import log as _mod_log
from ... import missing as _mod_missing
from ... import msgeditor, urlutils
from ...commands import Command
from ...i18n import gettext
from ...option import ListOption, Option, RegistryOption
from ...trace import note, warning
class cmd_forges(Command):
    __doc__ = 'List all known hosting sites and user details.'
    hidden = True

    def run(self):
        for instance in _mod_forge.iter_forge_instances():
            current_user = instance.get_current_user()
            if current_user is not None:
                current_user_url = instance.get_user_url(current_user)
                if current_user_url is not None:
                    self.outf.write(gettext('%s (%s) - user: %s (%s)\n') % (instance.name, instance.base_url, current_user, current_user_url))
                else:
                    self.outf.write(gettext('%s (%s) - user: %s\n') % (instance.name, instance.base_url, current_user))
            else:
                self.outf.write(gettext('%s (%s) - not logged in\n') % (instance.name, instance.base_url))