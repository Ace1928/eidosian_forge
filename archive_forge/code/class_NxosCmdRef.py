from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
class NxosCmdRef:
    """NXOS Command Reference utilities.
    The NxosCmdRef class takes a yaml-formatted string of nxos module commands
    and converts it into dict-formatted database of getters/setters/defaults
    and associated common and platform-specific values. The utility methods
    add additional data such as existing states, playbook states, and proposed cli.
    The utilities also abstract away platform differences such as different
    defaults and different command syntax.

    Callers must provide a yaml formatted string that defines each command and
    its properties; e.g. BFD global:
    ---
    _template: # _template holds common settings for all commands
      # Enable feature bfd if disabled
      feature: bfd
      # Common getter syntax for BFD commands
      get_command: show run bfd all | incl '^(no )*bfd'

    interval:
      kind: dict
      getval: bfd interval (?P<tx>\\d+) min_rx (?P<min_rx>\\d+) multiplier (?P<multiplier>\\d+)
      setval: bfd interval {tx} min_rx {min_rx} multiplier {multiplier}
      default:
        tx: 50
        min_rx: 50
        multiplier: 3
      N3K:
        # Platform overrides
        default:
          tx: 250
          min_rx: 250
          multiplier: 3
    """

    def __init__(self, module, cmd_ref_str, ref_only=False):
        """Initialize cmd_ref from yaml data."""
        self._module = module
        self._check_imports()
        self._yaml_load(cmd_ref_str)
        self.cache_existing = None
        self.present_states = ['present', 'merged', 'replaced']
        self.absent_states = ['absent', 'deleted']
        ref = self._ref
        ref['commands'] = sorted([k for k in ref if not k.startswith('_')])
        ref['_proposed'] = []
        ref['_context'] = []
        ref['_resource_key'] = None
        if not ref_only:
            ref['_state'] = module.params.get('state', 'present')
            self.feature_enable()
            self.get_platform_defaults()
            self.normalize_defaults()

    def __getitem__(self, key=None):
        if key is None:
            return self._ref
        return self._ref[key]

    def _check_imports(self):
        module = self._module
        msg = nxosCmdRef_import_check()
        if msg:
            module.fail_json(msg=msg)

    def _yaml_load(self, cmd_ref_str):
        if PY2:
            self._ref = yaml.load(cmd_ref_str)
        elif PY3:
            self._ref = yaml.load(cmd_ref_str, Loader=yaml.FullLoader)

    def feature_enable(self):
        """Add 'feature <foo>' to _proposed if ref includes a 'feature' key."""
        ref = self._ref
        feature = ref['_template'].get('feature')
        if feature:
            show_cmd = "show run | incl 'feature {0}'".format(feature)
            output = self.execute_show_command(show_cmd, 'text')
            if not output or 'CLI command error' in output:
                msg = "** 'feature {0}' is not enabled. Module will auto-enable feature {0} ** ".format(feature)
                self._module.warn(msg)
                ref['_proposed'].append('feature {0}'.format(feature))
                ref['_cli_is_feature_disabled'] = ref['_proposed']

    def get_platform_shortname(self):
        """Query device for platform type, normalize to a shortname/nickname.
        Returns platform shortname (e.g. 'N3K-3058P' returns 'N3K') or None.
        """
        platform_info = self.execute_show_command('show inventory', 'json')
        if not platform_info or not isinstance(platform_info, dict):
            return None
        inventory_table = platform_info['TABLE_inv']['ROW_inv']
        for info in inventory_table:
            if 'Chassis' in info['name']:
                network_os_platform = info['productid']
                break
        else:
            return None
        m = re.match('(?P<short>N[35679][K57])-(?P<N35>C35)*', network_os_platform)
        if not m:
            return None
        shortname = m.group('short')
        if m.groupdict().get('N35'):
            shortname = 'N35'
        elif re.match('N77', shortname):
            shortname = 'N7K'
        elif re.match('N3K|N9K', shortname):
            for info in inventory_table:
                if '-R' in info['productid']:
                    shortname += '-F'
                    break
        return shortname

    def get_platform_defaults(self):
        """Update ref with platform specific defaults"""
        plat = self.get_platform_shortname()
        if not plat:
            return
        ref = self._ref
        ref['_platform_shortname'] = plat
        for k in ref['commands']:
            if plat in ref[k].get('_exclude', ''):
                ref['commands'].remove(k)
        plat_spec_cmds = [k for k in ref['commands'] if plat in ref[k]]
        for k in plat_spec_cmds:
            for plat_key in ref[k][plat]:
                ref[k][plat_key] = ref[k][plat][plat_key]

    def normalize_defaults(self):
        """Update ref defaults with normalized data"""
        ref = self._ref
        for k in ref['commands']:
            if 'default' in ref[k] and ref[k]['default']:
                kind = ref[k]['kind']
                if 'int' == kind:
                    ref[k]['default'] = int(ref[k]['default'])
                elif 'list' == kind:
                    ref[k]['default'] = [str(i) for i in ref[k]['default']]
                elif 'dict' == kind:
                    for key, v in ref[k]['default'].items():
                        if v:
                            v = str(v)
                        ref[k]['default'][key] = v

    def execute_show_command(self, command, format):
        """Generic show command helper.
        Warning: 'CLI command error' exceptions are caught, must be handled by caller.
        Return device output as a newline-separated string or None.
        """
        cmds = [{'command': command, 'output': format}]
        output = None
        try:
            output = run_commands(self._module, cmds)
            if output:
                output = output[0]
        except ConnectionError as exc:
            if 'CLI command error' in repr(exc):
                output = repr(exc)
            else:
                raise
        return output

    def pattern_match_existing(self, output, k):
        """Pattern matching helper for `get_existing`.
        `k` is the command name string. Use the pattern from cmd_ref to
        find a matching string in the output.
        Return regex match object or None.
        """
        ref = self._ref
        pattern = re.compile(ref[k]['getval'])
        multiple = 'multiple' in ref[k].keys()
        match_lines = [re.search(pattern, line) for line in output]
        if 'dict' == ref[k]['kind']:
            match = [m for m in match_lines if m]
            if not match:
                return None
            if len(match) > 1 and (not multiple):
                raise ValueError('get_existing: multiple matches found for property {0}'.format(k))
        else:
            match = [m.groups() for m in match_lines if m]
            if not match:
                return None
            if len(match) > 1 and (not multiple):
                raise ValueError('get_existing: multiple matches found for property {0}'.format(k))
            for item in match:
                index = match.index(item)
                match[index] = list(item)
                if None is match[index][0]:
                    match[index].pop(0)
                elif 'no' in match[index][0]:
                    match[index].pop(0)
                    if not match:
                        return None
        return match

    def set_context(self, context=None):
        """Update ref with command context."""
        if context is None:
            context = []
        ref = self._ref
        ref['_context'] = ref['_template'].get('context', [])
        for cmd in context:
            ref['_context'].append(cmd)
        ref['_resource_key'] = context[-1] if context else ref['_resource_key']

    def get_existing(self, cache_output=None):
        """Update ref with existing command states from the device.
        Store these states in each command's 'existing' key.
        """
        ref = self._ref
        if ref.get('_cli_is_feature_disabled'):
            if ref['_state'] in self.present_states:
                [ref['_proposed'].append(ctx) for ctx in ref['_context']]
            return
        show_cmd = ref['_template']['get_command']
        if cache_output:
            output = cache_output
        else:
            output = self.execute_show_command(show_cmd, 'text') or []
            self.cache_existing = output
        if ref['_context']:
            output = CustomNetworkConfig(indent=2, contents=output)
            output = output.get_section(ref['_context'])
        if not output:
            if ref['_state'] in self.present_states:
                [ref['_proposed'].append(ctx) for ctx in ref['_context']]
            return
        if ref['_state'] in self.absent_states and ref['_context']:
            if ref['_resource_key'] and ref['_resource_key'] == ref['_context'][-1]:
                if ref['_context'][-1] in output:
                    ref['_context'][-1] = 'no ' + ref['_context'][-1]
                else:
                    del ref['_context'][-1]
                return
        output = output.split('\n')
        for k in ref['commands']:
            match = self.pattern_match_existing(output, k)
            if not match:
                continue
            ref[k]['existing'] = {}
            for item in match:
                index = match.index(item)
                kind = ref[k]['kind']
                if 'int' == kind:
                    ref[k]['existing'][index] = int(item[0])
                elif 'list' == kind:
                    ref[k]['existing'][index] = [str(i) for i in item[0]]
                elif 'dict' == kind:
                    ref[k]['existing'][index] = {}
                    for key in item.groupdict().keys():
                        ref[k]['existing'][index][key] = str(item.group(key))
                elif 'str' == kind:
                    ref[k]['existing'][index] = item[0]
                else:
                    raise ValueError("get_existing: unknown 'kind' value specified for key '{0}'".format(k))

    def get_playvals(self):
        """Update ref with values from the playbook.
        Store these values in each command's 'playval' key.
        """
        ref = self._ref
        module = self._module
        params = {}
        if module.params.get('config'):
            param_data = module.params.get('config')
            params['global'] = param_data
            for key in param_data.keys():
                if isinstance(param_data[key], list):
                    params[key] = param_data[key]
        else:
            params['global'] = module.params
        for k in ref.keys():
            for level in params.keys():
                if isinstance(params[level], dict):
                    params[level] = [params[level]]
                for item in params[level]:
                    if k in item and item[k] is not None:
                        if not ref[k].get('playval'):
                            ref[k]['playval'] = {}
                        playval = item[k]
                        index = params[level].index(item)
                        if 'int' == ref[k]['kind']:
                            playval = int(playval)
                        elif 'list' == ref[k]['kind']:
                            playval = [str(i) for i in playval]
                        elif 'dict' == ref[k]['kind']:
                            for key, v in playval.items():
                                playval[key] = str(v)
                        ref[k]['playval'][index] = playval

    def build_cmd_set(self, playval, existing, k):
        """Helper function to create list of commands to configure device
        Return a list of commands
        """
        ref = self._ref
        proposed = ref['_proposed']
        cmd = None
        kind = ref[k]['kind']
        if 'int' == kind:
            cmd = ref[k]['setval'].format(playval)
        elif 'list' == kind:
            cmd = ref[k]['setval'].format(*playval)
        elif 'dict' == kind:
            if ref[k]['setval'].startswith('path'):
                tmplt = 'path {name}'
                if 'depth' in playval:
                    tmplt += ' depth {depth}'
                if 'query_condition' in playval:
                    tmplt += ' query-condition {query_condition}'
                if 'filter_condition' in playval:
                    tmplt += ' filter-condition {filter_condition}'
                cmd = tmplt.format(**playval)
            else:
                cmd = ref[k]['setval'].format(**playval)
        elif 'str' == kind:
            if 'deleted' in str(playval):
                if existing:
                    cmd = 'no ' + ref[k]['setval'].format(existing)
            else:
                cmd = ref[k]['setval'].format(playval)
        else:
            raise ValueError("get_proposed: unknown 'kind' value specified for key '{0}'".format(k))
        if cmd:
            if ref['_state'] in self.absent_states and (not re.search('^no', cmd)):
                cmd = 'no ' + cmd
            [proposed.append(ctx) for ctx in ref['_context']]
            [proposed.append(ctx) for ctx in ref[k].get('context', [])]
            proposed.append(cmd)

    def get_proposed(self):
        """Compare playbook values against existing states and create a list
        of proposed commands.
        Return a list of raw cli command strings.
        """
        ref = self._ref
        proposed = ref['_proposed']
        if ref['_context'] and ref['_context'][-1].startswith('no'):
            [proposed.append(ctx) for ctx in ref['_context']]
            return proposed
        play_keys = [k for k in ref['commands'] if 'playval' in ref[k]]

        def compare(playval, existing):
            if ref['_state'] in self.present_states:
                if existing is None:
                    return False
                elif str(playval) == str(existing):
                    return True
                elif isinstance(existing, dict) and playval in existing.values():
                    return True
            if ref['_state'] in self.absent_states:
                if isinstance(existing, dict) and all((x is None for x in existing.values())):
                    existing = None
                if existing is None or playval not in existing.values():
                    return True
            return False
        for k in play_keys:
            playval = ref[k]['playval']
            playval_copy = deepcopy(playval)
            existing = ref[k].get('existing', ref[k]['default'])
            multiple = 'multiple' in ref[k].keys()
            if isinstance(existing, dict) and multiple:
                for ekey, evalue in existing.items():
                    if isinstance(evalue, dict):
                        evalue = dict(((k, v) for k, v in evalue.items() if v != 'None'))
                    for pkey, pvalue in playval.items():
                        if compare(pvalue, evalue):
                            if playval_copy.get(pkey):
                                del playval_copy[pkey]
                if not playval_copy:
                    continue
            else:
                for pkey, pval in playval.items():
                    if compare(pval, existing):
                        if playval_copy.get(pkey):
                            del playval_copy[pkey]
                if not playval_copy:
                    continue
            playval = playval_copy
            if isinstance(existing, dict):
                for dkey, dvalue in existing.items():
                    for pval in playval.values():
                        self.build_cmd_set(pval, dvalue, k)
            else:
                for pval in playval.values():
                    self.build_cmd_set(pval, existing, k)
        cmds = sorted(set(proposed), key=lambda x: proposed.index(x))
        return cmds