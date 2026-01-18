import json
import sys
import textwrap
import traceback
import enum
from oslo_config import cfg
import prettytable
from oslo_upgradecheck._i18n import _
class UpgradeCommands(object):
    """Base class for upgrade checks

    This class should be inherited by a class in each project that provides
    the actual checks. Those checks should be added to the _upgrade_checks
    class member so that they are run when the ``check`` method is called.

    The subcommands here must not rely on the service object model since they
    should be able to run on n-1 data. Any queries to the database should be
    done through the sqlalchemy query language directly like the database
    schema migrations.
    """
    display_title = _('Upgrade Check Results')
    _upgrade_checks = ()

    def _get_details(self, upgrade_check_result):
        if upgrade_check_result.details is not None:
            return '\n'.join(textwrap.wrap(upgrade_check_result.details, 60, subsequent_indent='  '))

    def check(self):
        """Performs checks to see if the deployment is ready for upgrade.

        These checks are expected to be run BEFORE services are restarted with
        new code.

        :returns: Code
        """
        return_code = Code.SUCCESS
        check_results = []
        for name, func in self._upgrade_checks:
            if isinstance(func, tuple):
                func_name, kwargs = func
                result = func_name(self, **kwargs)
            else:
                result = func(self)
            check_results.append((name, result))
            if result.code > return_code:
                return_code = result.code
        if hasattr(CONF, 'command') and hasattr(CONF.command, 'json') and CONF.command.json:
            output = {'name': str(self.display_title), 'checks': []}
            for name, result in check_results:
                output['checks'].append({'check': name, 'result': result.code, 'details': result.details})
            print(json.dumps(output))
        else:
            t = prettytable.PrettyTable([str(self.display_title)], hrules=prettytable.ALL)
            t.align = 'l'
            for name, result in check_results:
                cell = _('Check: %(name)s\nResult: %(result)s\nDetails: %(details)s') % {'name': name, 'result': UPGRADE_CHECK_MSG_MAP[result.code], 'details': self._get_details(result)}
                t.add_row([cell])
            print(t)
        return return_code