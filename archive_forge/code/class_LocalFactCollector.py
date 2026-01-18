from __future__ import (absolute_import, division, print_function)
import glob
import json
import os
import stat
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import configparser, StringIO
class LocalFactCollector(BaseFactCollector):
    name = 'local'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        local_facts = {}
        local_facts['local'] = {}
        if not module:
            return local_facts
        fact_path = module.params.get('fact_path', None)
        if not fact_path or not os.path.exists(fact_path):
            return local_facts
        local = {}
        for fn in sorted(glob.glob(fact_path + '/*.fact')):
            fact_base = os.path.basename(fn).replace('.fact', '')
            failed = None
            try:
                executable_fact = stat.S_IXUSR & os.stat(fn)[stat.ST_MODE]
            except OSError as e:
                failed = 'Could not stat fact (%s): %s' % (fn, to_text(e))
                local[fact_base] = failed
                module.warn(failed)
                continue
            if executable_fact:
                try:
                    rc, out, err = module.run_command(fn)
                    if rc != 0:
                        failed = 'Failure executing fact script (%s), rc: %s, err: %s' % (fn, rc, err)
                except (IOError, OSError) as e:
                    failed = 'Could not execute fact script (%s): %s' % (fn, to_text(e))
                if failed is not None:
                    local[fact_base] = failed
                    module.warn(failed)
                    continue
            else:
                out = get_file_content(fn, default='')
            try:
                out = to_text(out, errors='surrogate_or_strict')
            except UnicodeError:
                fact = 'error loading fact - output of running "%s" was not utf-8' % fn
                local[fact_base] = fact
                module.warn(fact)
                continue
            try:
                fact = json.loads(out)
            except ValueError:
                cp = configparser.ConfigParser()
                try:
                    if PY3:
                        cp.read_file(StringIO(out))
                    else:
                        cp.readfp(StringIO(out))
                except configparser.Error:
                    fact = 'error loading facts as JSON or ini - please check content: %s' % fn
                    module.warn(fact)
                else:
                    fact = {}
                    for sect in cp.sections():
                        if sect not in fact:
                            fact[sect] = {}
                        for opt in cp.options(sect):
                            val = cp.get(sect, opt)
                            fact[sect][opt] = val
            except Exception as e:
                fact = 'Failed to convert (%s) to JSON: %s' % (fn, to_text(e))
                module.warn(fact)
            local[fact_base] = fact
        local_facts['local'] = local
        return local_facts