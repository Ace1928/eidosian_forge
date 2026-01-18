import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _complete_token_pparam(self, text, path, command, pparams, kparams):
    """
        Completes a positional parameter token, which can also be the keywork
        part of a kparam token, as before the '=' sign is on the line, the
        parser cannot know better.
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
    completions = []
    target = self._current_node.get_node(path)
    cmd_params, free_pparams, free_kparams = target.get_command_signature(command)
    current_parameters = {}
    for index in range(len(pparams)):
        if index < len(cmd_params):
            current_parameters[cmd_params[index]] = pparams[index]
    for key, value in six.iteritems(kparams):
        current_parameters[key] = value
    self._completion_help_topic = command
    completion_method = target.get_completion_method(command)
    self.log.debug('Command %s accepts parameters %s.' % (command, cmd_params))
    pparam_ok = True
    for index in range(len(cmd_params)):
        param = cmd_params[index]
        if param in kparams:
            if index <= len(pparams):
                pparam_ok = False
                self.log.debug('No more possible pparams (because of kparams).')
                break
        elif text.strip() == '' and len(pparams) == len(cmd_params) or len(pparams) > len(cmd_params):
            pparam_ok = False
            self.log.debug('No more possible pparams.')
            break
    else:
        if len(cmd_params) == 0:
            pparam_ok = False
            self.log.debug('No more possible pparams (none exists)')
    if pparam_ok:
        if not text:
            pparam_index = len(pparams)
        else:
            pparam_index = len(pparams) - 1
        self._current_parameter = cmd_params[pparam_index]
        self.log.debug('Completing pparam %s.' % self._current_parameter)
        if completion_method:
            pparam_completions = completion_method(current_parameters, text, self._current_parameter)
            if pparam_completions is not None:
                completions.extend(pparam_completions)
    if text:
        offset = 1
    else:
        offset = 0
    keyword_completions = [param + '=' for param in cmd_params[len(pparams) - offset:] if param not in kparams if param.startswith(text)]
    self.log.debug('Possible pparam values are %s.' % str(completions))
    self.log.debug('Possible kparam keywords are %s.' % str(keyword_completions))
    if keyword_completions:
        if self._current_parameter:
            self._current_token = self.con.render_text(self._current_parameter, self.prefs['color_parameter']) + '|' + self.con.render_text('keyword=', self.prefs['color_keyword'])
        else:
            self._current_token = self.con.render_text('keyword=', self.prefs['color_keyword'])
    elif self._current_parameter:
        self._current_token = self.con.render_text(self._current_parameter, self.prefs['color_parameter'])
    else:
        self._current_token = ''
    completions.extend(keyword_completions)
    if free_kparams or free_pparams:
        self.log.debug('Command has free [kp]params.')
        if completion_method:
            self.log.debug('Calling completion method for free params.')
            free_completions = completion_method(current_parameters, text, '*')
            do_free_pparams = False
            do_free_kparams = False
            for free_completion in free_completions:
                if free_completion.endswith('='):
                    do_free_kparams = True
                else:
                    do_free_pparams = True
            if do_free_pparams:
                self._current_token = self.con.render_text(free_pparams, self.prefs['color_parameter']) + '|' + self._current_token
                self._current_token = self._current_token.rstrip('|')
                if not self._current_parameter:
                    self._current_parameter = 'free_parameter'
            if do_free_kparams:
                if not 'keyword=' in self._current_token:
                    self._current_token = self.con.render_text('keyword=', self.prefs['color_keyword']) + '|' + self._current_token
                    self._current_token = self._current_token.rstrip('|')
                if not self._current_parameter:
                    self._current_parameter = 'free_parameter'
            completions.extend(free_completions)
    self.log.debug('Found completions %s.' % str(completions))
    return completions