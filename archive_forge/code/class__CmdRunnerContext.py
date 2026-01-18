from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
class _CmdRunnerContext(object):

    def __init__(self, runner, args_order, output_process, ignore_value_none, check_mode_skip, check_mode_return, **kwargs):
        self.runner = runner
        self.args_order = tuple(args_order)
        self.output_process = output_process
        self.ignore_value_none = ignore_value_none
        self.check_mode_skip = check_mode_skip
        self.check_mode_return = check_mode_return
        self.run_command_args = dict(kwargs)
        self.environ_update = runner.environ_update
        self.environ_update.update(self.run_command_args.get('environ_update', {}))
        if runner.force_lang:
            self.environ_update.update({'LANGUAGE': runner.force_lang, 'LC_ALL': runner.force_lang})
        self.run_command_args['environ_update'] = self.environ_update
        if 'check_rc' not in self.run_command_args:
            self.run_command_args['check_rc'] = runner.check_rc
        self.check_rc = self.run_command_args['check_rc']
        self.cmd = None
        self.results_rc = None
        self.results_out = None
        self.results_err = None
        self.results_processed = None

    def run(self, **kwargs):
        runner = self.runner
        module = self.runner.module
        self.cmd = list(runner.command)
        self.context_run_args = dict(kwargs)
        named_args = dict(module.params)
        named_args.update(kwargs)
        for arg_name in self.args_order:
            value = None
            try:
                if arg_name in named_args:
                    value = named_args[arg_name]
                elif not runner.arg_formats[arg_name].ignore_missing_value:
                    raise MissingArgumentValue(self.args_order, arg_name)
                self.cmd.extend(runner.arg_formats[arg_name](value, ctx_ignore_none=self.ignore_value_none))
            except MissingArgumentValue:
                raise
            except Exception as e:
                raise FormatError(arg_name, value, runner.arg_formats[arg_name], e)
        if self.check_mode_skip and module.check_mode:
            return self.check_mode_return
        results = module.run_command(self.cmd, **self.run_command_args)
        self.results_rc, self.results_out, self.results_err = results
        self.results_processed = self.output_process(*results)
        return self.results_processed

    @property
    def run_info(self):
        return dict(ignore_value_none=self.ignore_value_none, check_rc=self.check_rc, environ_update=self.environ_update, args_order=self.args_order, cmd=self.cmd, run_command_args=self.run_command_args, context_run_args=self.context_run_args, results_rc=self.results_rc, results_out=self.results_out, results_err=self.results_err, results_processed=self.results_processed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False