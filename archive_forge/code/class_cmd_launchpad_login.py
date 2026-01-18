from ... import branch as _mod_branch
from ... import controldir, trace
from ...commands import Command
from ...errors import CommandError, NotBranchError
from ...i18n import gettext
from ...option import ListOption, Option
class cmd_launchpad_login(Command):
    __doc__ = "Show or set the Launchpad user ID.\n\n    When communicating with Launchpad, some commands need to know your\n    Launchpad user ID.  This command can be used to set or show the\n    user ID that Bazaar will use for such communication.\n\n    :Examples:\n      Show the Launchpad ID of the current user::\n\n          brz launchpad-login\n\n      Set the Launchpad ID of the current user to 'bob'::\n\n          brz launchpad-login bob\n    "
    aliases = ['lp-login']
    takes_args = ['name?']
    takes_options = ['verbose', Option('no-check', "Don't check that the user name is valid."), Option('service-root', type=str, help='Launchpad service root to connect to')]

    def run(self, name=None, no_check=False, verbose=False, service_root='production'):
        from . import account
        check_account = not no_check
        if name is None:
            username = account.get_lp_login()
            if username:
                if check_account:
                    account.check_lp_login(username)
                    if verbose:
                        self.outf.write(gettext('Launchpad user ID exists and has SSH keys.\n'))
                self.outf.write(username + '\n')
            else:
                self.outf.write(gettext('No Launchpad user ID configured.\n'))
                return 1
        else:
            name = name.lower()
            if check_account:
                account.check_lp_login(name)
                if verbose:
                    self.outf.write(gettext('Launchpad user ID exists and has SSH keys.\n'))
            account.set_lp_login(name)
            if verbose:
                self.outf.write(gettext("Launchpad user ID set to '%s'.\n") % (name,))
        if check_account:
            from .lp_api import connect_launchpad
            from .uris import lookup_service_root
            connect_launchpad(lookup_service_root(service_root))