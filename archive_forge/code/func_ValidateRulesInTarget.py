import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def ValidateRulesInTarget(target, target_dict, extra_sources_for_rules):
    """Ensures that the rules sections in target_dict are valid and consistent,
  and determines which sources they apply to.

  Arguments:
    target: string, name of target.
    target_dict: dict, target spec containing "rules" and "sources" lists.
    extra_sources_for_rules: a list of keys to scan for rule matches in
        addition to 'sources'.
  """
    rule_names = {}
    rule_extensions = {}
    rules = target_dict.get('rules', [])
    for rule in rules:
        rule_name = rule['rule_name']
        if rule_name in rule_names:
            raise GypError(f'rule {rule_name} exists in duplicate, target {target}')
        rule_names[rule_name] = rule
        rule_extension = rule['extension']
        if rule_extension.startswith('.'):
            rule_extension = rule_extension[1:]
        if rule_extension in rule_extensions:
            raise GypError(('extension %s associated with multiple rules, ' + 'target %s rules %s and %s') % (rule_extension, target, rule_extensions[rule_extension]['rule_name'], rule_name))
        rule_extensions[rule_extension] = rule
        if 'rule_sources' in rule:
            raise GypError('rule_sources must not exist in input, target %s rule %s' % (target, rule_name))
        rule_sources = []
        source_keys = ['sources']
        source_keys.extend(extra_sources_for_rules)
        for source_key in source_keys:
            for source in target_dict.get(source_key, []):
                source_root, source_extension = os.path.splitext(source)
                if source_extension.startswith('.'):
                    source_extension = source_extension[1:]
                if source_extension == rule_extension:
                    rule_sources.append(source)
        if len(rule_sources) > 0:
            rule['rule_sources'] = rule_sources