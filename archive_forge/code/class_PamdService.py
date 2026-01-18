from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
class PamdService(object):

    def __init__(self, content):
        self._head = None
        self._tail = None
        for line in content.splitlines():
            if line.lstrip().startswith('#'):
                pamd_line = PamdComment(line)
            elif line.lstrip().startswith('@include'):
                pamd_line = PamdInclude(line)
            elif line.strip() == '':
                pamd_line = PamdEmptyLine(line)
            else:
                pamd_line = PamdRule.rule_from_string(line)
            self.append(pamd_line)

    def append(self, pamd_line):
        if self._head is None:
            self._head = self._tail = pamd_line
        else:
            pamd_line.prev = self._tail
            pamd_line.next = None
            self._tail.next = pamd_line
            self._tail = pamd_line

    def remove(self, rule_type, rule_control, rule_path):
        current_line = self._head
        changed = 0
        while current_line is not None:
            if current_line.matches(rule_type, rule_control, rule_path):
                if current_line.prev is not None:
                    current_line.prev.next = current_line.next
                    if current_line.next is not None:
                        current_line.next.prev = current_line.prev
                else:
                    self._head = current_line.next
                    current_line.next.prev = None
                changed += 1
            current_line = current_line.next
        return changed

    def get(self, rule_type, rule_control, rule_path):
        lines = []
        current_line = self._head
        while current_line is not None:
            if isinstance(current_line, PamdRule) and current_line.matches(rule_type, rule_control, rule_path):
                lines.append(current_line)
            current_line = current_line.next
        return lines

    def has_rule(self, rule_type, rule_control, rule_path):
        if self.get(rule_type, rule_control, rule_path):
            return True
        return False

    def update_rule(self, rule_type, rule_control, rule_path, new_type=None, new_control=None, new_path=None, new_args=None):
        rules_to_find = self.get(rule_type, rule_control, rule_path)
        new_args = parse_module_arguments(new_args, return_none=True)
        changes = 0
        for current_rule in rules_to_find:
            rule_changed = False
            if new_type:
                if current_rule.rule_type != new_type:
                    rule_changed = True
                    current_rule.rule_type = new_type
            if new_control:
                if current_rule.rule_control != new_control:
                    rule_changed = True
                    current_rule.rule_control = new_control
            if new_path:
                if current_rule.rule_path != new_path:
                    rule_changed = True
                    current_rule.rule_path = new_path
            if new_args is not None:
                if current_rule.rule_args != new_args:
                    rule_changed = True
                    current_rule.rule_args = new_args
            if rule_changed:
                changes += 1
        return changes

    def insert_before(self, rule_type, rule_control, rule_path, new_type=None, new_control=None, new_path=None, new_args=None):
        rules_to_find = self.get(rule_type, rule_control, rule_path)
        changes = 0
        for current_rule in rules_to_find:
            new_rule = PamdRule(new_type, new_control, new_path, new_args)
            previous_rule = current_rule.prev
            while previous_rule is not None and isinstance(previous_rule, (PamdComment, PamdEmptyLine)):
                previous_rule = previous_rule.prev
            if previous_rule is not None and (not previous_rule.matches(new_type, new_control, new_path)):
                previous_rule.next = new_rule
                new_rule.prev = previous_rule
                new_rule.next = current_rule
                current_rule.prev = new_rule
                changes += 1
            elif previous_rule is None:
                if current_rule.prev is None:
                    self._head = new_rule
                else:
                    current_rule.prev.next = new_rule
                new_rule.prev = current_rule.prev
                new_rule.next = current_rule
                current_rule.prev = new_rule
                changes += 1
        return changes

    def insert_after(self, rule_type, rule_control, rule_path, new_type=None, new_control=None, new_path=None, new_args=None):
        rules_to_find = self.get(rule_type, rule_control, rule_path)
        changes = 0
        for current_rule in rules_to_find:
            next_rule = current_rule.next
            while next_rule is not None and isinstance(next_rule, (PamdComment, PamdEmptyLine)):
                next_rule = next_rule.next
            new_rule = PamdRule(new_type, new_control, new_path, new_args)
            if next_rule is not None and (not next_rule.matches(new_type, new_control, new_path)):
                next_rule.prev = new_rule
                new_rule.next = next_rule
                new_rule.prev = current_rule
                current_rule.next = new_rule
                changes += 1
            elif next_rule is None:
                new_rule.prev = self._tail
                new_rule.next = None
                self._tail.next = new_rule
                self._tail = new_rule
                current_rule.next = new_rule
                changes += 1
        return changes

    def add_module_arguments(self, rule_type, rule_control, rule_path, args_to_add):
        rules_to_find = self.get(rule_type, rule_control, rule_path)
        args_to_add = parse_module_arguments(args_to_add)
        changes = 0
        for current_rule in rules_to_find:
            rule_changed = False
            simple_new_args = set()
            key_value_new_args = dict()
            for arg in args_to_add:
                if arg.startswith('['):
                    continue
                elif '=' in arg:
                    key, value = arg.split('=')
                    key_value_new_args[key] = value
                else:
                    simple_new_args.add(arg)
            key_value_new_args_set = set(key_value_new_args)
            simple_current_args = set()
            key_value_current_args = dict()
            for arg in current_rule.rule_args:
                if arg.startswith('['):
                    continue
                elif '=' in arg:
                    key, value = arg.split('=')
                    key_value_current_args[key] = value
                else:
                    simple_current_args.add(arg)
            key_value_current_args_set = set(key_value_current_args)
            new_args_to_add = list()
            if simple_new_args.difference(simple_current_args):
                for arg in simple_new_args.difference(simple_current_args):
                    new_args_to_add.append(arg)
            if key_value_new_args_set.difference(key_value_current_args_set):
                for key in key_value_new_args_set.difference(key_value_current_args_set):
                    new_args_to_add.append(key + '=' + key_value_new_args[key])
            if new_args_to_add:
                current_rule.rule_args += new_args_to_add
                rule_changed = True
            if key_value_new_args_set.intersection(key_value_current_args_set):
                for key in key_value_new_args_set.intersection(key_value_current_args_set):
                    if key_value_current_args[key] != key_value_new_args[key]:
                        arg_index = current_rule.rule_args.index(key + '=' + key_value_current_args[key])
                        current_rule.rule_args[arg_index] = str(key + '=' + key_value_new_args[key])
                        rule_changed = True
            if rule_changed:
                changes += 1
        return changes

    def remove_module_arguments(self, rule_type, rule_control, rule_path, args_to_remove):
        rules_to_find = self.get(rule_type, rule_control, rule_path)
        args_to_remove = parse_module_arguments(args_to_remove)
        changes = 0
        for current_rule in rules_to_find:
            if not args_to_remove:
                args_to_remove = []
            if not list(set(current_rule.rule_args) & set(args_to_remove)):
                continue
            current_rule.rule_args = [arg for arg in current_rule.rule_args if arg not in args_to_remove]
            changes += 1
        return changes

    def validate(self):
        current_line = self._head
        while current_line is not None:
            curr_validate = current_line.validate()
            if not curr_validate[0]:
                return curr_validate
            current_line = current_line.next
        return (True, 'Module is valid')

    def __str__(self):
        lines = []
        current_line = self._head
        mark = '# Updated by Ansible - %s' % datetime.now().isoformat()
        while current_line is not None:
            lines.append(str(current_line))
            current_line = current_line.next
        if len(lines) <= 1:
            lines.insert(0, '')
            lines.insert(1, mark)
        elif lines[1].startswith('# Updated by Ansible'):
            lines[1] = mark
        else:
            lines.insert(1, mark)
        return '\n'.join(lines) + '\n'