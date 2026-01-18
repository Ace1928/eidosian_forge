from __future__ import absolute_import, division, print_function
import getpass
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from functools import wraps
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import cache_loader, cliconf_loader, connection_loader, terminal_loader
from ansible_collections.ansible.netcommon.plugins.connection.libssh import HAS_PYLIBSSH
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
def _handle_prompt(self, resp, prompts, answer, newline, prompt_retry_check=False, check_all=False):
    """
        Matches the command prompt and responds

        :arg resp: Byte string containing the raw response from the remote
        :arg prompts: Sequence of byte strings that we consider prompts for input
        :arg answer: Sequence of Byte string to send back to the remote if we find a prompt.
                A carriage return is automatically appended to this string.
        :param prompt_retry_check: Bool value for trying to detect more prompts
        :param check_all: Bool value to indicate if all the values in prompt sequence should be matched or any one of
                          given prompt.
        :returns: True if a prompt was found in ``resp``. If check_all is True
                  will True only after all the prompt in the prompts list are matched. False otherwise.
        """
    single_prompt = False
    if not isinstance(prompts, list):
        prompts = [prompts]
        single_prompt = True
    if not isinstance(answer, list):
        answer = [answer]
    try:
        prompts_regex = [re.compile(to_bytes(r), re.I) for r in prompts]
    except re.error as exc:
        raise ConnectionError('Failed to compile one or more terminal prompt regexes: %s.\nPrompts provided: %s' % (to_text(exc), prompts))
    for index, regex in enumerate(prompts_regex):
        match = regex.search(resp)
        if match:
            self._matched_cmd_prompt = match.group()
            self._log_messages('matched command prompt: %s' % self._matched_cmd_prompt)
            if not prompt_retry_check:
                prompt_answer = to_bytes(answer[index] if len(answer) > index else answer[0])
                if newline:
                    prompt_answer += b'\r'
                self._ssh_shell.sendall(prompt_answer)
                self._log_messages('matched command prompt answer: %s' % prompt_answer)
            if check_all and prompts and (not single_prompt):
                prompts.pop(0)
                answer.pop(0)
                return False
            return True
    return False