import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def _process_args_to_opts(self, args_that_override: Optional[List[str]]=None):
    self.opt = Opt(vars(self.args))
    extra_ag = []
    if '_subparser' in self.opt:
        self.overridable.update(self.opt['_subparser'].overridable)
        extra_ag = self.opt.pop('_subparser')._action_groups
    self.opt['parlai_home'] = self.parlai_home
    self.opt = self._infer_datapath(self.opt)
    option_strings_dict = {}
    store_true = []
    store_false = []
    for group in self._action_groups + extra_ag:
        for a in group._group_actions:
            if hasattr(a, 'option_strings'):
                for option in a.option_strings:
                    option_strings_dict[option] = a.dest
                    if isinstance(a, argparse._StoreTrueAction):
                        store_true.append(option)
                    elif isinstance(a, argparse._StoreFalseAction):
                        store_false.append(option)
    if args_that_override is None:
        args_that_override = _sys.argv[1:]
    args_that_override = fix_underscores(args_that_override)
    for i in range(len(args_that_override)):
        if args_that_override[i] in option_strings_dict:
            if args_that_override[i] in store_true:
                self.overridable[option_strings_dict[args_that_override[i]]] = True
            elif args_that_override[i] in store_false:
                self.overridable[option_strings_dict[args_that_override[i]]] = False
            elif i < len(args_that_override) - 1 and args_that_override[i + 1] not in option_strings_dict:
                key = option_strings_dict[args_that_override[i]]
                self.overridable[key] = self.opt[key]
    self.opt['override'] = self.overridable
    if self.opt.get('init_opt', None) is not None:
        self._load_opts(self.opt)
    options_to_change = {'model_file', 'dict_file', 'bpe_vocab', 'bpe_merge'}
    for each_key in options_to_change:
        if self.opt.get(each_key) is not None:
            self.opt[each_key] = modelzoo_path(self.opt.get('datapath'), self.opt[each_key])
        if self.opt['override'].get(each_key) is not None:
            self.opt['override'][each_key] = modelzoo_path(self.opt.get('datapath'), self.opt['override'][each_key])
    self.opt['starttime'] = datetime.datetime.today().strftime('%b%d_%H-%M')