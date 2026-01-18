import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
class _Parse:
    """A helper class that parsing main arguments of `CCompilerOpt`,
    also parsing configuration statements in dispatch-able sources.

    Parameters
    ----------
    cpu_baseline : str or None
        minimal set of required CPU features or special options.

    cpu_dispatch : str or None
        dispatched set of additional CPU features or special options.

    Special options can be:
        - **MIN**: Enables the minimum CPU features that utilized via `_Config.conf_min_features`
        - **MAX**: Enables all supported CPU features by the Compiler and platform.
        - **NATIVE**: Enables all CPU features that supported by the current machine.
        - **NONE**: Enables nothing
        - **Operand +/-**: remove or add features, useful with options **MAX**, **MIN** and **NATIVE**.
            NOTE: operand + is only added for nominal reason.

    NOTES:
        - Case-insensitive among all CPU features and special options.
        - Comma or space can be used as a separator.
        - If the CPU feature is not supported by the user platform or compiler,
          it will be skipped rather than raising a fatal error.
        - Any specified CPU features to 'cpu_dispatch' will be skipped if its part of CPU baseline features
        - 'cpu_baseline' force enables implied features.

    Attributes
    ----------
    parse_baseline_names : list
        Final CPU baseline's feature names(sorted from low to high)
    parse_baseline_flags : list
        Compiler flags of baseline features
    parse_dispatch_names : list
        Final CPU dispatch-able feature names(sorted from low to high)
    parse_target_groups : dict
        Dictionary containing initialized target groups that configured
        through class attribute `conf_target_groups`.

        The key is represent the group name and value is a tuple
        contains three items :
            - bool, True if group has the 'baseline' option.
            - list, list of CPU features.
            - list, list of extra compiler flags.

    """

    def __init__(self, cpu_baseline, cpu_dispatch):
        self._parse_policies = dict(KEEP_BASELINE=(None, self._parse_policy_not_keepbase, []), KEEP_SORT=(self._parse_policy_keepsort, self._parse_policy_not_keepsort, []), MAXOPT=(self._parse_policy_maxopt, None, []), WERROR=(self._parse_policy_werror, None, []), AUTOVEC=(self._parse_policy_autovec, None, ['MAXOPT']))
        if hasattr(self, 'parse_is_cached'):
            return
        self.parse_baseline_names = []
        self.parse_baseline_flags = []
        self.parse_dispatch_names = []
        self.parse_target_groups = {}
        if self.cc_noopt:
            cpu_baseline = cpu_dispatch = None
        self.dist_log('check requested baseline')
        if cpu_baseline is not None:
            cpu_baseline = self._parse_arg_features('cpu_baseline', cpu_baseline)
            baseline_names = self.feature_names(cpu_baseline)
            self.parse_baseline_flags = self.feature_flags(baseline_names)
            self.parse_baseline_names = self.feature_sorted(self.feature_implies_c(baseline_names))
        self.dist_log('check requested dispatch-able features')
        if cpu_dispatch is not None:
            cpu_dispatch_ = self._parse_arg_features('cpu_dispatch', cpu_dispatch)
            cpu_dispatch = {f for f in cpu_dispatch_ if f not in self.parse_baseline_names}
            conflict_baseline = cpu_dispatch_.difference(cpu_dispatch)
            self.parse_dispatch_names = self.feature_sorted(self.feature_names(cpu_dispatch))
            if len(conflict_baseline) > 0:
                self.dist_log('skip features', conflict_baseline, 'since its part of baseline')
        self.dist_log('initialize targets groups')
        for group_name, tokens in self.conf_target_groups.items():
            self.dist_log('parse target group', group_name)
            GROUP_NAME = group_name.upper()
            if not tokens or not tokens.strip():
                self.parse_target_groups[GROUP_NAME] = (False, [], [])
                continue
            has_baseline, features, extra_flags = self._parse_target_tokens(tokens)
            self.parse_target_groups[GROUP_NAME] = (has_baseline, features, extra_flags)
        self.parse_is_cached = True

    def parse_targets(self, source):
        """
        Fetch and parse configuration statements that required for
        defining the targeted CPU features, statements should be declared
        in the top of source in between **C** comment and start
        with a special mark **@targets**.

        Configuration statements are sort of keywords representing
        CPU features names, group of statements and policies, combined
        together to determine the required optimization.

        Parameters
        ----------
        source : str
            the path of **C** source file.

        Returns
        -------
        - bool, True if group has the 'baseline' option
        - list, list of CPU features
        - list, list of extra compiler flags
        """
        self.dist_log("looking for '@targets' inside -> ", source)
        with open(source) as fd:
            tokens = ''
            max_to_reach = 1000
            start_with = '@targets'
            start_pos = -1
            end_with = '*/'
            end_pos = -1
            for current_line, line in enumerate(fd):
                if current_line == max_to_reach:
                    self.dist_fatal('reached the max of lines')
                    break
                if start_pos == -1:
                    start_pos = line.find(start_with)
                    if start_pos == -1:
                        continue
                    start_pos += len(start_with)
                tokens += line
                end_pos = line.find(end_with)
                if end_pos != -1:
                    end_pos += len(tokens) - len(line)
                    break
        if start_pos == -1:
            self.dist_fatal("expected to find '%s' within a C comment" % start_with)
        if end_pos == -1:
            self.dist_fatal("expected to end with '%s'" % end_with)
        tokens = tokens[start_pos:end_pos]
        return self._parse_target_tokens(tokens)
    _parse_regex_arg = re.compile('\\s|,|([+-])')

    def _parse_arg_features(self, arg_name, req_features):
        if not isinstance(req_features, str):
            self.dist_fatal("expected a string in '%s'" % arg_name)
        final_features = set()
        tokens = list(filter(None, re.split(self._parse_regex_arg, req_features)))
        append = True
        for tok in tokens:
            if tok[0] in ('#', '$'):
                self.dist_fatal(arg_name, "target groups and policies aren't allowed from arguments, only from dispatch-able sources")
            if tok == '+':
                append = True
                continue
            if tok == '-':
                append = False
                continue
            TOK = tok.upper()
            features_to = set()
            if TOK == 'NONE':
                pass
            elif TOK == 'NATIVE':
                native = self.cc_flags['native']
                if not native:
                    self.dist_fatal(arg_name, "native option isn't supported by the compiler")
                features_to = self.feature_names(force_flags=native, macros=[('DETECT_FEATURES', 1)])
            elif TOK == 'MAX':
                features_to = self.feature_supported.keys()
            elif TOK == 'MIN':
                features_to = self.feature_min
            elif TOK in self.feature_supported:
                features_to.add(TOK)
            elif not self.feature_is_exist(TOK):
                self.dist_fatal(arg_name, ", '%s' isn't a known feature or option" % tok)
            if append:
                final_features = final_features.union(features_to)
            else:
                final_features = final_features.difference(features_to)
            append = True
        return final_features
    _parse_regex_target = re.compile('\\s|[*,/]|([()])')

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

    def _parse_token_policy(self, token):
        """validate policy token"""
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'$' must stuck in the begin of policy name")
        token = token[1:]
        if token not in self._parse_policies:
            self.dist_fatal("'%s' is an invalid policy name, available policies are" % token, self._parse_policies.keys())
        return token

    def _parse_token_group(self, token, has_baseline, final_targets, extra_flags):
        """validate group token"""
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'#' must stuck in the begin of group name")
        token = token[1:]
        ghas_baseline, gtargets, gextra_flags = self.parse_target_groups.get(token, (False, None, []))
        if gtargets is None:
            self.dist_fatal("'%s' is an invalid target group name, " % token + 'available target groups are', self.parse_target_groups.keys())
        if ghas_baseline:
            has_baseline = True
        final_targets += [f for f in gtargets if f not in final_targets]
        extra_flags += [f for f in gextra_flags if f not in extra_flags]
        return (has_baseline, final_targets, extra_flags)

    def _parse_multi_target(self, targets):
        """validate multi targets that defined between parentheses()"""
        if not targets:
            self.dist_fatal("empty multi-target '()'")
        if not all([self.feature_is_exist(tar) for tar in targets]):
            self.dist_fatal('invalid target name in multi-target', targets)
        if not all([tar in self.parse_baseline_names or tar in self.parse_dispatch_names for tar in targets]):
            return None
        targets = self.feature_ahead(targets)
        if not targets:
            return None
        targets = self.feature_sorted(targets)
        targets = tuple(targets)
        return targets

    def _parse_policy_not_keepbase(self, has_baseline, final_targets, extra_flags):
        """skip all baseline features"""
        skipped = []
        for tar in final_targets[:]:
            is_base = False
            if isinstance(tar, str):
                is_base = tar in self.parse_baseline_names
            else:
                is_base = all([f in self.parse_baseline_names for f in tar])
            if is_base:
                skipped.append(tar)
                final_targets.remove(tar)
        if skipped:
            self.dist_log('skip baseline features', skipped)
        return (has_baseline, final_targets, extra_flags)

    def _parse_policy_keepsort(self, has_baseline, final_targets, extra_flags):
        """leave a notice that $keep_sort is on"""
        self.dist_log("policy 'keep_sort' is on, dispatch-able targets", final_targets, "\nare 'not' sorted depend on the highest interest butas specified in the dispatch-able source or the extra group")
        return (has_baseline, final_targets, extra_flags)

    def _parse_policy_not_keepsort(self, has_baseline, final_targets, extra_flags):
        """sorted depend on the highest interest"""
        final_targets = self.feature_sorted(final_targets, reverse=True)
        return (has_baseline, final_targets, extra_flags)

    def _parse_policy_maxopt(self, has_baseline, final_targets, extra_flags):
        """append the compiler optimization flags"""
        if self.cc_has_debug:
            self.dist_log("debug mode is detected, policy 'maxopt' is skipped.")
        elif self.cc_noopt:
            self.dist_log("optimization is disabled, policy 'maxopt' is skipped.")
        else:
            flags = self.cc_flags['opt']
            if not flags:
                self.dist_log("current compiler doesn't support optimization flags, policy 'maxopt' is skipped", stderr=True)
            else:
                extra_flags += flags
        return (has_baseline, final_targets, extra_flags)

    def _parse_policy_werror(self, has_baseline, final_targets, extra_flags):
        """force warnings to treated as errors"""
        flags = self.cc_flags['werror']
        if not flags:
            self.dist_log("current compiler doesn't support werror flags, warnings will 'not' treated as errors", stderr=True)
        else:
            self.dist_log('compiler warnings are treated as errors')
            extra_flags += flags
        return (has_baseline, final_targets, extra_flags)

    def _parse_policy_autovec(self, has_baseline, final_targets, extra_flags):
        """skip features that has no auto-vectorized support by compiler"""
        skipped = []
        for tar in final_targets[:]:
            if isinstance(tar, str):
                can = self.feature_can_autovec(tar)
            else:
                can = all([self.feature_can_autovec(t) for t in tar])
            if not can:
                final_targets.remove(tar)
                skipped.append(tar)
        if skipped:
            self.dist_log('skip non auto-vectorized features', skipped)
        return (has_baseline, final_targets, extra_flags)