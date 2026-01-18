import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
class Checker(FirstPartyCaveatChecker):
    """ Holds a set of checkers for first party caveats.
    """

    def __init__(self, namespace=None, include_std_checkers=True):
        if namespace is None:
            namespace = Namespace()
        self._namespace = namespace
        self._checkers = {}
        if include_std_checkers:
            self.register_std()

    def check_first_party_caveat(self, ctx, cav):
        """ Checks the caveat against all registered caveat conditions.
        :return: error message string if any or None
        """
        try:
            cond, arg = parse_caveat(cav)
        except ValueError as ex:
            return 'cannot parse caveat "{}": {}'.format(cav, ex.args[0])
        checker = self._checkers.get(cond)
        if checker is None:
            return 'caveat "{}" not satisfied: caveat not recognized'.format(cav)
        err = checker.check(ctx, cond, arg)
        if err is not None:
            return 'caveat "{}" not satisfied: {}'.format(cav, err)

    def namespace(self):
        """ Returns the namespace associated with the Checker.
        """
        return self._namespace

    def info(self):
        """ Returns information on all the registered checkers.

        Sorted by namespace and then name
        :returns a list of CheckerInfo
        """
        return sorted(self._checkers.values(), key=lambda x: (x.ns, x.name))

    def register(self, cond, uri, check):
        """ Registers the given condition(string) in the given namespace
        uri (string) to be checked with the given check function.
        The check function checks a caveat by passing an auth context, a cond
        parameter(string) that holds the caveat condition including any
        namespace prefix and an arg parameter(string) that hold any additional
        caveat argument text. It will return any error as string otherwise
        None.

        It will raise a ValueError if the namespace is not registered or
        if the condition has already been registered.
        """
        if check is None:
            raise RegisterError('no check function registered for namespace {} when registering condition {}'.format(uri, cond))
        prefix = self._namespace.resolve(uri)
        if prefix is None:
            raise RegisterError('no prefix registered for namespace {} when registering condition {}'.format(uri, cond))
        if prefix == '' and cond.find(':') >= 0:
            raise RegisterError('caveat condition {} in namespace {} contains a colon but its prefix is empty'.format(cond, uri))
        full_cond = condition_with_prefix(prefix, cond)
        info = self._checkers.get(full_cond)
        if info is not None:
            raise RegisterError('checker for {} (namespace {}) already registered in namespace {}'.format(full_cond, uri, info.ns))
        self._checkers[full_cond] = CheckerInfo(check=check, ns=uri, name=cond, prefix=prefix)

    def register_std(self):
        """ Registers all the standard checkers in the given checker.

        If not present already, the standard checkers schema (STD_NAMESPACE) is
        added to the checker's namespace with an empty prefix.
        """
        self._namespace.register(STD_NAMESPACE, '')
        for cond in _ALL_CHECKERS:
            self.register(cond, STD_NAMESPACE, _ALL_CHECKERS[cond])