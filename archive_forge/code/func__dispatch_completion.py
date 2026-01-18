import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _dispatch_completion(self, path, command, pparams, kparams, text, current_token):
    """
        This method takes care of dispatching the current completion request
        from readline (via the _complete() method) to the relevant token
        completion methods. It has to cope with the fact that the commandline
        being incomplete yet,
        Of course, as the command line is still unfinished, the parser can
        only do so much of a job. For instance, until the '=' sign is on the
        command line, there is no way to distinguish a positional parameter
        from the begining of a keyword=value parameter.
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
        @param current_token: Name of token to complete.
        @type current_token: str
        @return: Possible completions for the token.
        @rtype: list of str
        """
    completions = []
    self.log.debug('Dispatching completion for %s token. ' % current_token + "text='%s', path='%s', command='%s', " % (text, path, command) + 'pparams=%s, kparams=%s' % (str(pparams), str(kparams)))
    path, iterall = path.partition('*')[:2]
    if iterall:
        try:
            target = self._current_node.get_node(path)
        except ValueError:
            cpl_path = path
        else:
            children = target.children
            if children:
                cpl_path = children[0].path
    else:
        cpl_path = path
    if current_token == 'command':
        completions = self._complete_token_command(text, cpl_path, command)
    elif current_token == 'path':
        completions = self._complete_token_path(text)
    elif current_token == 'pparam':
        completions = self._complete_token_pparam(text, cpl_path, command, pparams, kparams)
    elif current_token == 'kparam':
        completions = self._complete_token_kparam(text, cpl_path, command, pparams, kparams)
    else:
        self.log.debug('Cannot complete unknown token %s.' % current_token)
    return completions