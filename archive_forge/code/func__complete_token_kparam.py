import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _complete_token_kparam(self, text, path, command, pparams, kparams):
    """
        Completes a keyword=value parameter token.
        @param path: Path of the target ConfigNode.
        @type path: str
        @param command: The command (if any) found by the parser.
        @type command: str
        @param pparams: Positional parameters from commandline.
        @type pparams: list of str
        @param kparams: Keyword parameters from commandline.
        @type kparams: dict of str:str
        @param text: Current text being typed by the user.
        @type text: str
        @return: Possible completions for the token.
        @rtype: list of str
        """
    self.log.debug("Called for text='%s'" % text)
    target = self._current_node.get_node(path)
    cmd_params = target.get_command_signature(command)[0]
    self.log.debug('Command %s accepts parameters %s.' % (command, cmd_params))
    keyword, sep, current_value = text.partition('=')
    self.log.debug("Completing '%s' for kparam %s" % (current_value, keyword))
    self._current_parameter = keyword
    current_parameters = {}
    for index in range(len(pparams)):
        current_parameters[cmd_params[index]] = pparams[index]
    for key, value in six.iteritems(kparams):
        current_parameters[key] = value
    completion_method = target.get_completion_method(command)
    if completion_method:
        completions = completion_method(current_parameters, current_value, keyword)
        if completions is None:
            completions = []
    self._current_token = self.con.render_text(self._current_parameter, self.prefs['color_parameter'])
    self.log.debug('Found completions %s.' % str(completions))
    return ['%s=%s' % (keyword, completion) for completion in completions]