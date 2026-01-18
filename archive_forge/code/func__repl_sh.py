import os.path
import signal
import sys
import pexpect
def _repl_sh(command, args, non_printable_insert):
    child = pexpect.spawn(command, args, echo=False, encoding='utf-8')
    ps1 = PEXPECT_PROMPT[:5] + non_printable_insert + PEXPECT_PROMPT[5:]
    ps2 = PEXPECT_CONTINUATION_PROMPT[:5] + non_printable_insert + PEXPECT_CONTINUATION_PROMPT[5:]
    prompt_change = u"PS1='{0}' PS2='{1}' PROMPT_COMMAND=''".format(ps1, ps2)
    return REPLWrapper(child, u'\\$', prompt_change, extra_init_cmd='export PAGER=cat')