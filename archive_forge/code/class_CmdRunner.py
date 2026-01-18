from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
class CmdRunner(object):
    """
    Wrapper for ``AnsibleModule.run_command()``.

    It aims to provide a reusable runner with consistent argument formatting
    and sensible defaults.
    """

    @staticmethod
    def _prepare_args_order(order):
        return tuple(order) if is_sequence(order) else tuple(order.split())

    def __init__(self, module, command, arg_formats=None, default_args_order=(), check_rc=False, force_lang='C', path_prefix=None, environ_update=None):
        self.module = module
        self.command = _ensure_list(command)
        self.default_args_order = self._prepare_args_order(default_args_order)
        if arg_formats is None:
            arg_formats = {}
        self.arg_formats = dict(arg_formats)
        self.check_rc = check_rc
        self.force_lang = force_lang
        self.path_prefix = path_prefix
        if environ_update is None:
            environ_update = {}
        self.environ_update = environ_update
        _cmd = self.command[0]
        self.command[0] = _cmd if os.path.isabs(_cmd) or '/' in _cmd else module.get_bin_path(_cmd, opt_dirs=path_prefix, required=True)
        for mod_param_name, spec in iteritems(module.argument_spec):
            if mod_param_name not in self.arg_formats:
                self.arg_formats[mod_param_name] = _Format.as_default_type(spec.get('type', 'str'), mod_param_name)

    @property
    def binary(self):
        return self.command[0]

    def __call__(self, args_order=None, output_process=None, ignore_value_none=True, check_mode_skip=False, check_mode_return=None, **kwargs):
        if output_process is None:
            output_process = _process_as_is
        if args_order is None:
            args_order = self.default_args_order
        args_order = self._prepare_args_order(args_order)
        for p in args_order:
            if p not in self.arg_formats:
                raise MissingArgumentFormat(p, args_order, tuple(self.arg_formats.keys()))
        return _CmdRunnerContext(runner=self, args_order=args_order, output_process=output_process, ignore_value_none=ignore_value_none, check_mode_skip=check_mode_skip, check_mode_return=check_mode_return, **kwargs)

    def has_arg_format(self, arg):
        return arg in self.arg_formats
    context = __call__