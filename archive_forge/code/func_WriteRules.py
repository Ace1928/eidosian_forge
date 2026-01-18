import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteRules(self, rules, extra_sources, extra_outputs, extra_mac_bundle_resources, part_of_all):
    """Write Makefile code for any 'rules' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       rules (used to make other pieces dependent on these rules)
        part_of_all: flag indicating this target is part of 'all'
        """
    env = self.GetSortedXcodeEnv()
    for rule in rules:
        name = StringToMakefileVariable('{}_{}'.format(self.qualified_target, rule['rule_name']))
        count = 0
        self.WriteLn('### Generated for rule %s:' % name)
        all_outputs = []
        for rule_source in rule.get('rule_sources', []):
            dirs = set()
            rule_source_dirname, rule_source_basename = os.path.split(rule_source)
            rule_source_root, rule_source_ext = os.path.splitext(rule_source_basename)
            outputs = [self.ExpandInputRoot(out, rule_source_root, rule_source_dirname) for out in rule['outputs']]
            for out in outputs:
                dir = os.path.dirname(out)
                if dir:
                    dirs.add(dir)
            if int(rule.get('process_outputs_as_sources', False)):
                extra_sources += outputs
            if int(rule.get('process_outputs_as_mac_bundle_resources', False)):
                extra_mac_bundle_resources += outputs
            inputs = [Sourceify(self.Absolutify(i)) for i in [rule_source] + rule.get('inputs', [])]
            actions = ['$(call do_cmd,%s_%d)' % (name, count)]
            if name == 'resources_grit':
                actions += ['@touch --no-create $@']
            outputs = [gyp.xcode_emulation.ExpandEnvVars(o, env) for o in outputs]
            inputs = [gyp.xcode_emulation.ExpandEnvVars(i, env) for i in inputs]
            outputs = [self.Absolutify(o) for o in outputs]
            all_outputs += outputs
            self.WriteLn('%s: obj := $(abs_obj)' % outputs[0])
            self.WriteLn('%s: builddir := $(abs_builddir)' % outputs[0])
            self.WriteMakeRule(outputs, inputs, actions, command='%s_%d' % (name, count))
            variables_with_spaces = re.compile('\\$\\([^ ]* \\$<\\)')
            for output in outputs:
                output = re.sub(variables_with_spaces, '', output)
                assert ' ' not in output, 'Spaces in rule filenames not yet supported (%s)' % output
            self.WriteLn('all_deps += %s' % ' '.join(outputs))
            action = [self.ExpandInputRoot(ac, rule_source_root, rule_source_dirname) for ac in rule['action']]
            mkdirs = ''
            if len(dirs) > 0:
                mkdirs = 'mkdir -p %s; ' % ' '.join(dirs)
            cd_action = 'cd %s; ' % Sourceify(self.path or '.')
            if self.flavor == 'mac':
                action = [gyp.xcode_emulation.ExpandEnvVars(command, env) for command in action]
            action = gyp.common.EncodePOSIXShellList(action)
            action = action.replace('$(TARGET)', self.target)
            cd_action = cd_action.replace('$(TARGET)', self.target)
            mkdirs = mkdirs.replace('$(TARGET)', self.target)
            self.WriteLn('cmd_%(name)s_%(count)d = LD_LIBRARY_PATH=$(builddir)/lib.host:$(builddir)/lib.target:$$LD_LIBRARY_PATH; export LD_LIBRARY_PATH; %(cd_action)s%(mkdirs)s%(action)s' % {'action': action, 'cd_action': cd_action, 'count': count, 'mkdirs': mkdirs, 'name': name})
            self.WriteLn('quiet_cmd_%(name)s_%(count)d = RULE %(name)s_%(count)d $@' % {'count': count, 'name': name})
            self.WriteLn()
            count += 1
        outputs_variable = 'rule_%s_outputs' % name
        self.WriteList(all_outputs, outputs_variable)
        extra_outputs.append('$(%s)' % outputs_variable)
        self.WriteLn('### Finished generating for rule: %s' % name)
        self.WriteLn()
    self.WriteLn('### Finished generating for all rules')
    self.WriteLn('')