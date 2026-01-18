import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_target_tokens(self, tokens):
    assert isinstance(tokens, str)
    final_targets = []
    extra_flags = []
    has_baseline = False
    skipped = set()
    policies = set()
    multi_target = None
    tokens = list(filter(None, re.split(self._parse_regex_target, tokens)))
    if not tokens:
        self.dist_fatal('expected one token at least')
    for tok in tokens:
        TOK = tok.upper()
        ch = tok[0]
        if ch in ('+', '-'):
            self.dist_fatal("+/- are 'not' allowed from target's groups or @targets, only from cpu_baseline and cpu_dispatch parms")
        elif ch == '$':
            if multi_target is not None:
                self.dist_fatal("policies aren't allowed inside multi-target '()', only CPU features")
            policies.add(self._parse_token_policy(TOK))
        elif ch == '#':
            if multi_target is not None:
                self.dist_fatal("target groups aren't allowed inside multi-target '()', only CPU features")
            has_baseline, final_targets, extra_flags = self._parse_token_group(TOK, has_baseline, final_targets, extra_flags)
        elif ch == '(':
            if multi_target is not None:
                self.dist_fatal("unclosed multi-target, missing ')'")
            multi_target = set()
        elif ch == ')':
            if multi_target is None:
                self.dist_fatal("multi-target opener '(' wasn't found")
            targets = self._parse_multi_target(multi_target)
            if targets is None:
                skipped.add(tuple(multi_target))
            else:
                if len(targets) == 1:
                    targets = targets[0]
                if targets and targets not in final_targets:
                    final_targets.append(targets)
            multi_target = None
        else:
            if TOK == 'BASELINE':
                if multi_target is not None:
                    self.dist_fatal("baseline isn't allowed inside multi-target '()'")
                has_baseline = True
                continue
            if multi_target is not None:
                multi_target.add(TOK)
                continue
            if not self.feature_is_exist(TOK):
                self.dist_fatal("invalid target name '%s'" % TOK)
            is_enabled = TOK in self.parse_baseline_names or TOK in self.parse_dispatch_names
            if is_enabled:
                if TOK not in final_targets:
                    final_targets.append(TOK)
                continue
            skipped.add(TOK)
    if multi_target is not None:
        self.dist_fatal("unclosed multi-target, missing ')'")
    if skipped:
        self.dist_log('skip targets', skipped, 'not part of baseline or dispatch-able features')
    final_targets = self.feature_untied(final_targets)
    for p in list(policies):
        _, _, deps = self._parse_policies[p]
        for d in deps:
            if d in policies:
                continue
            self.dist_log("policy '%s' force enables '%s'" % (p, d))
            policies.add(d)
    for p, (have, nhave, _) in self._parse_policies.items():
        func = None
        if p in policies:
            func = have
            self.dist_log("policy '%s' is ON" % p)
        else:
            func = nhave
        if not func:
            continue
        has_baseline, final_targets, extra_flags = func(has_baseline, final_targets, extra_flags)
    return (has_baseline, final_targets, extra_flags)