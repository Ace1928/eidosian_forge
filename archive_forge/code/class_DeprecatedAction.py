import argparse
import logging
import os
import sys
from keystoneauth1 import loading
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client
from novaclient import exceptions as exc
import novaclient.extension
from novaclient.i18n import _
from novaclient import utils
class DeprecatedAction(argparse.Action):
    """An argparse action for deprecated options.

    This class is an ``argparse.Action`` subclass that allows command
    line options to be explicitly deprecated.  It modifies the help
    text for the option to indicate that it's deprecated (unless help
    has been suppressed using ``argparse.SUPPRESS``), and provides a
    means to specify an alternate option to use using the ``use``
    keyword argument to ``argparse.ArgumentParser.add_argument()``.
    The original action may be specified with the ``real_action``
    keyword argument, which has the same interpretation as the
    ``action`` argument to ``argparse.ArgumentParser.add_argument()``,
    with the addition of the special "nothing" action which completely
    ignores the option (other than emitting the deprecation warning).
    Note that the deprecation warning is only emitted once per
    specific option string.

    Note: If the ``real_action`` keyword argument specifies an unknown
    action, no warning will be emitted unless the action is used, due
    to limitations with the method used to resolve the action names.
    """

    def __init__(self, option_strings, dest, help=None, real_action=None, use=None, **kwargs):
        """Initialize a ``DeprecatedAction`` instance.

        :param option_strings: The recognized option strings.
        :param dest: The attribute that will be set.
        :param help: Help text.  This will be updated to indicate the
                     deprecation, and if ``use`` is provided, that
                     text will be included as well.
        :param real_action: The actual action to invoke.  This is
                            interpreted the same way as the ``action``
                            parameter.
        :param use: Text explaining which option to use instead.
        """
        if not help:
            if use:
                help = _('Deprecated; %(use)s') % {'use': use}
            else:
                help = _('Deprecated')
        elif help != argparse.SUPPRESS:
            if use:
                help = _('%(help)s (Deprecated; %(use)s)') % {'help': help, 'use': use}
            else:
                help = _('%(help)s (Deprecated)') % {'help': help}
        super(DeprecatedAction, self).__init__(option_strings, dest, help=help, **kwargs)
        self.emitted = set()
        self.use = use
        if real_action == 'nothing':
            self.real_action_args = False
            self.real_action = None
        elif real_action is None or isinstance(real_action, str):
            self.real_action_args = (option_strings, dest, help, kwargs)
            self.real_action = real_action
        else:
            self.real_action_args = False
            self.real_action = real_action(option_strings, dest, help=help, **kwargs)

    def _get_action(self, parser):
        """Retrieve the action callable.

        This internal method is used to retrieve the callable
        implementing the action.  If ``real_action`` was specified as
        ``None`` or one of the standard string names, an internal
        method of the ``argparse.ArgumentParser`` instance is used to
        resolve it into an actual action class, which is then
        instantiated.  This is cached, in case the action is called
        multiple times.

        :param parser: The ``argparse.ArgumentParser`` instance.

        :returns: The action callable.
        """
        if self.real_action_args is not False:
            option_strings, dest, help, kwargs = self.real_action_args
            action_class = parser._registry_get('action', self.real_action)
            if action_class is None:
                print(_('WARNING: Programming error: Unknown real action "%s"') % self.real_action, file=sys.stderr)
                self.real_action = None
            else:
                self.real_action = action_class(option_strings, dest, help=help, **kwargs)
            self.real_action_args = False
        return self.real_action

    def __call__(self, parser, namespace, values, option_string):
        """Implement the action.

        Emits the deprecation warning message (only once for any given
        option string), then calls the real action (if any).

        :param parser: The ``argparse.ArgumentParser`` instance.
        :param namespace: The ``argparse.Namespace`` object which
                          should have an attribute set.
        :param values: Any arguments provided to the option.
        :param option_string: The option string that was used.
        """
        action = self._get_action(parser)
        if option_string not in self.emitted:
            if self.use:
                print(_('WARNING: Option "%(option)s" is deprecated; %(use)s') % {'option': option_string, 'use': self.use}, file=sys.stderr)
            else:
                print(_('WARNING: Option "%(option)s" is deprecated') % {'option': option_string}, file=sys.stderr)
            self.emitted.add(option_string)
        if action:
            action(parser, namespace, values, option_string)