import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteMakeRule(self, outputs, inputs, actions=None, comment=None, order_only=False, force=False, phony=False, command=None):
    """Write a Makefile rule, with some extra tricks.

        outputs: a list of outputs for the rule (note: this is not directly
                 supported by make; see comments below)
        inputs: a list of inputs for the rule
        actions: a list of shell commands to run for the rule
        comment: a comment to put in the Makefile above the rule (also useful
                 for making this Python script's code self-documenting)
        order_only: if true, makes the dependency order-only
        force: if true, include FORCE_DO_CMD as an order-only dep
        phony: if true, the rule does not actually generate the named output, the
               output is just a name to run the rule
        command: (optional) command name to generate unambiguous labels
        """
    outputs = [QuoteSpaces(o) for o in outputs]
    inputs = [QuoteSpaces(i) for i in inputs]
    if comment:
        self.WriteLn('# ' + comment)
    if phony:
        self.WriteLn('.PHONY: ' + ' '.join(outputs))
    if actions:
        self.WriteLn('%s: TOOLSET := $(TOOLSET)' % outputs[0])
    force_append = ' FORCE_DO_CMD' if force else ''
    if order_only:
        self.WriteLn('{}: | {}{}'.format(' '.join(outputs), ' '.join(inputs), force_append))
    elif len(outputs) == 1:
        self.WriteLn('{}: {}{}'.format(outputs[0], ' '.join(inputs), force_append))
    else:
        cmddigest = hashlib.sha1((command or self.target).encode('utf-8')).hexdigest()
        intermediate = '%s.intermediate' % cmddigest
        self.WriteLn('{}: {}'.format(' '.join(outputs), intermediate))
        self.WriteLn('\t%s' % '@:')
        self.WriteLn('{}: {}'.format('.INTERMEDIATE', intermediate))
        self.WriteLn('{}: {}{}'.format(intermediate, ' '.join(inputs), force_append))
        actions.insert(0, '$(call do_cmd,touch)')
    if actions:
        for action in actions:
            self.WriteLn('\t%s' % action)
    self.WriteLn()